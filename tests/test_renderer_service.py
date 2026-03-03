from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

from services.renderer import main as renderer_main


def test_renderer_pdf_smoke(monkeypatch) -> None:
    def _fake_run(command, cwd=None, capture_output=False, text=False, check=False):  # noqa: ANN001
        Path(cwd or '.').joinpath('main.pdf').write_bytes(b'%PDF-1.4\n% mock renderer pdf\n')
        return subprocess.CompletedProcess(args=command, returncode=0, stdout='ok', stderr='')

    monkeypatch.setattr(renderer_main.subprocess, 'run', _fake_run)
    payload = renderer_main.RenderPDFRequest(
        tex='\\documentclass{article}\\begin{document}Hello\\end{document}',
        assets={},
    )
    response = asyncio.run(renderer_main.render_pdf(payload))

    assert response.status_code == 200
    assert response.body.startswith(b'%PDF')
    assert response.headers.get('content-type', '').startswith('application/pdf')
