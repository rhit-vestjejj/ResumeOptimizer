from __future__ import annotations

import base64
import binascii
from pathlib import Path
import subprocess
import tempfile
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title='Resume Renderer', version='1.0.0')


class RenderPDFRequest(BaseModel):
    tex: str
    assets: Dict[str, str] = Field(default_factory=dict)


def _safe_asset_name(raw: str) -> str:
    cleaned = (raw or '').strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail='Asset filename cannot be empty.')
    name = Path(cleaned).name
    if name != cleaned:
        raise HTTPException(status_code=400, detail=f'Invalid asset filename: {cleaned}')
    return name


@app.get('/healthz')
async def healthcheck() -> Dict[str, str]:
    return {'status': 'ok'}


@app.post('/render/pdf')
async def render_pdf(payload: RenderPDFRequest) -> Response:
    tex = (payload.tex or '').strip()
    if not tex:
        raise HTTPException(status_code=400, detail='Field "tex" is required.')

    try:
        with tempfile.TemporaryDirectory(prefix='resume-renderer-', dir='/tmp') as tmp_dir:
            workdir = Path(tmp_dir)
            tex_path = workdir / 'main.tex'
            tex_path.write_text(payload.tex, encoding='utf-8')

            for raw_name, encoded in payload.assets.items():
                name = _safe_asset_name(raw_name)
                try:
                    decoded = base64.b64decode((encoded or '').encode('utf-8'), validate=True)
                except (binascii.Error, ValueError):
                    raise HTTPException(status_code=400, detail=f'Invalid base64 payload for asset: {name}')
                (workdir / name).write_bytes(decoded)

            command = [
                'latexmk',
                '-pdf',
                '-interaction=nonstopmode',
                '-halt-on-error',
                'main.tex',
            ]
            result = subprocess.run(
                command,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'LaTeX compile failed.',
                        'stdout': (result.stdout or '')[-4000:],
                        'stderr': (result.stderr or '')[-4000:],
                    },
                )

            pdf_path = workdir / 'main.pdf'
            if not pdf_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'LaTeX compile did not produce main.pdf.',
                        'stdout': (result.stdout or '')[-4000:],
                        'stderr': (result.stderr or '')[-4000:],
                    },
                )

            pdf_bytes = pdf_path.read_bytes()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Renderer runtime error: {exc}')

    return Response(content=pdf_bytes, media_type='application/pdf')
