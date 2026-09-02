from pathlib import Path


DOCKER_ROOT = Path(__file__).resolve().parent


def test_build_script_prepares_the_isolated_viewer_bundle() -> None:
    script = (DOCKER_ROOT / "build.sh").read_text(encoding="utf-8")

    assert 'VIEWER_WEB_DIR="$SCRIPT_DIR/viewer_web"' in script
    assert 'npm --prefix "$VIEWER_WEB_DIR" ci --offline --ignore-scripts' in script
    assert 'npm --prefix "$VIEWER_WEB_DIR" run build' in script
    assert 'cp -a "$VIEWER_WEB_DIR/dist" "$APP_CONTEXT/viewer"' in script


def test_image_and_api_publish_the_isolated_viewer_at_viewer() -> None:
    dockerfile = (DOCKER_ROOT / "Dockerfile").read_text(encoding="utf-8")
    api = (DOCKER_ROOT / "api.py").read_text(encoding="utf-8")
    vite_config = (DOCKER_ROOT / "viewer_web" / "vite.config.ts").read_text(
        encoding="utf-8"
    )

    assert "COPY --from=app viewer /app/viewer" in dockerfile
    assert "from fastapi.staticfiles import StaticFiles" in api
    assert 'app.mount("/viewer", StaticFiles(directory="/app/viewer", html=True, check_dir=False), name="viewer")' in api
    assert 'base: "./"' in vite_config
