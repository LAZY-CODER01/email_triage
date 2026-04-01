from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Email Triage Environment is running!"}

@app.post("/reset")
def reset_env():
    return {"message": "Environment reset successfully"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()