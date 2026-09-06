from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("mineral_prediction_api.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
