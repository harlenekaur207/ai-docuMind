from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, get_service
from app.models import DocumentInfo, Source


class FakeRAG:
    def __init__(self):
        self.documents = []

    def index_document(self, path: Path, original_name: str):
        document = DocumentInfo(id="abc123", filename=original_name, pages=1, chunks=1)
        self.documents.append(document)
        return document

    def list_documents(self):
        return self.documents

    def delete_document(self, document_id: str):
        before = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.id != document_id]
        return len(self.documents) < before

    def answer(self, question: str):
        if not self.documents:
            raise ValueError("Upload and index at least one document before asking a question.")
        return "Employees receive 15 days of paid vacation [1].", [
            Source(reference=1, filename="handbook.pdf", page=12, excerpt="Employees receive 15 days...")
        ]


fake = FakeRAG()
app.dependency_overrides[get_service] = lambda: fake
client = TestClient(app)


def setup_function():
    fake.documents = []


def test_health():
    assert client.get("/health").status_code == 200


def test_rejects_non_pdf():
    response = client.post("/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_list_ask_and_delete():
    upload = client.post(
        "/documents/upload", files={"file": ("handbook.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )
    assert upload.status_code == 201
    assert client.get("/documents").json()[0]["filename"] == "handbook.pdf"
    answer = client.post("/ask", json={"question": "What is the vacation policy?"})
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["page"] == 12
    assert client.delete("/documents/abc123").status_code == 204


def test_ask_without_documents():
    response = client.post("/ask", json={"question": "What is the policy?"})
    assert response.status_code == 400

