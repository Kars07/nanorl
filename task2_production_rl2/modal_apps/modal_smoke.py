import modal

app = modal.App("task2-modal-control-plane-smoke")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), timeout=60)
def smoke() -> str:
    return "modal-ok"


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())

