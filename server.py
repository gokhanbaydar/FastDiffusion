from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from core import sdxl_styles, generator
import uvicorn

app = FastAPI(debug=True)
fd = generator.FastDiffusionGenerator()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def error_response(message):
    return JSONResponse(content={"message": message, "last_process_time": fd.proc_elapsed_time})


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    styles = sdxl_styles.style_keys
    context = {
        "request": request,
        "styles": styles,
        "last_process_time": fd.proc_elapsed_time,
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/load_models", response_class=JSONResponse)
async def load_models(request: Request):
    fd.load_models()
    return JSONResponse(content={"message": "Models loaded", "last_process_time": fd.proc_elapsed_time})


@app.get("/generate", response_class=JSONResponse)
async def generate(request: Request):
    if not fd.models_loaded:
        return error_response("Models not loaded")
    pp = request.query_params.get("pprompt")
    np = request.query_params.get("nprompt")
    st = request.query_params.get("style")
    st = sdxl_styles.styles.get(st)
    print(st)
    pp = st[0].replace("{prompt}", pp)
    np = np + "," + st[1]

    return JSONResponse(
        content={
            "message": "Image Generated",
            "prompt": pp,
            "negative_prompt": np,
            "style": request.query_params.get("style"),
            # "image": fd.image,
            "last_process_time": fd.proc_elapsed_time,
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
