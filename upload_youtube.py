#!/usr/bin/env python3
"""Subida explícita a YouTube con OAuth persistente y reintentos resumibles."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True); parser.add_argument("--title", required=True); parser.add_argument("--description", default="")
    parser.add_argument("--tags", default=""); parser.add_argument("--privacy", choices=("private", "unlisted", "public"), default="private")
    parser.add_argument("--client-secrets", required=True); parser.add_argument("--token", default="token.json")
    parser.add_argument("--thumbnail"); parser.add_argument("--playlist"); parser.add_argument("--subtitles")
    parser.add_argument("--language", default="es"); parser.add_argument("--publish-at", help="ISO 8601 UTC; requiere privacidad private.")
    parser.add_argument("--draft", action="store_true", help="Prepara y muestra metadatos sin subir nada.")
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024, help="Bytes por fragmento resumible (múltiplo de 256 KiB).")
    args = parser.parse_args(); video, secrets, token = Path(args.video), Path(args.client_secrets), Path(args.token)
    if args.chunk_size < 256 * 1024 or args.chunk_size % (256 * 1024):
        parser.error("--chunk-size debe ser múltiplo de 256 KiB.")
    if not video.is_file(): sys.exit(f"[ERROR] No existe el vídeo: {video}")
    if not secrets.is_file(): sys.exit("[ERROR] No existe el archivo de credenciales OAuth.")
    body = {"snippet": {"title": args.title, "description": args.description, "tags": [x.strip() for x in args.tags.split(",") if x.strip()], "categoryId": "24", "defaultLanguage": args.language}, "status": {"privacyStatus": args.privacy, "selfDeclaredMadeForKids": False}}
    if args.publish_at:
        if args.privacy != "private": sys.exit("[ERROR] --publish-at requiere --privacy private.")
        body["status"]["publishAt"] = args.publish_at
    if args.draft:
        print(json.dumps({"video": str(video), "metadata": body, "thumbnail": args.thumbnail, "playlist": args.playlist, "subtitles": args.subtitles}, ensure_ascii=False, indent=2)); return
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError: sys.exit("[ERROR] Instala requirements-youtube.txt.")
    credentials = None
    if token.is_file():
        try: credentials = Credentials.from_authorized_user_file(str(token), SCOPES)
        except Exception: print("[AVISO] token OAuth no utilizable; se solicitará inicio de sesión.")
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        credentials = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES).run_local_server(port=0)
    token.write_text(credentials.to_json(), encoding="utf-8")
    try: token.chmod(0o600)
    except OSError: pass
    youtube = build("youtube", "v3", credentials=credentials)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=MediaFileUpload(str(video), mimetype="video/mp4", resumable=True, chunksize=args.chunk_size))
    response = None
    retries = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            retries = 0
            if status:
                print(f"Subida: {status.progress() * 100:.1f}%")
        except Exception as exc:
            retries += 1
            if retries > 5: sys.exit(f"[ERROR] La subida falló tras reintentos: {exc}")
            wait = 2 ** (retries - 1)
            print(f"[AVISO] Error de red; reintento {retries}/5 en {wait}s: {exc}", file=sys.stderr)
            time.sleep(wait)
    video_id = response["id"]
    if args.thumbnail:
        thumbnail = Path(args.thumbnail)
        if not thumbnail.is_file(): sys.exit(f"[ERROR] Thumbnail inexistente: {thumbnail}")
        youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumbnail))).execute()
    if args.playlist:
        youtube.playlistItems().insert(part="snippet", body={"snippet": {"playlistId": args.playlist, "resourceId": {"kind": "youtube#video", "videoId": video_id}}}).execute()
    if args.subtitles:
        subtitles = Path(args.subtitles)
        if not subtitles.is_file(): sys.exit(f"[ERROR] Subtítulos inexistentes: {subtitles}")
        youtube.captions().insert(part="snippet", body={"snippet": {"videoId": video_id, "language": args.language, "name": "Español", "isDraft": False}}, media_body=MediaFileUpload(str(subtitles), mimetype="application/octet-stream")).execute()
    print(f"Subido: https://www.youtube.com/watch?v={video_id}")


if __name__ == "__main__": main()
