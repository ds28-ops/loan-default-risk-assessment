from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Loan Doc Reader", description="Reads uploaded loan text files.")

@app.post("/read_txt", response_class=PlainTextResponse)
async def read_txt(file: UploadFile = File(...)):
    content = await file.read()
    return content.decode("utf-8")