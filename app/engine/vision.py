from __future__ import annotations

import os

SYSTEM_PROMPT = (
    "Você é um assistente especializado em descrever imagens extraídas de PDFs. "
    "Descreva o conteúdo da imagem de forma objetiva e detalhada em português."
)

USER_PROMPT = (
    "Descreva detalhadamente esta imagem extraída de um PDF jurídico/didático. "
    "Inclua: tipo de conteúdo (gráfico, tabela, diagrama, fotografia, esquema), "
    "elementos visuais principais, e texto relevante visível na imagem. "
    "Se a imagem for muito pequena ou parecer um ícone decorativo, informe isso."
)

_FALLBACK_MODELS = [
    os.getenv("OPENROUTER_VISION_MODEL"),
    "qwen/qwen-2-vl-72b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "mistralai/pixtral-12b:free",
    "openrouter/free",
]


def describe_image(
    image_data: str,
    image_format: str = "png",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://motor-pdf-production.up.railway.app",
                "X-Title": "Motor PDF",
            },
        )

        mime = f"image/{image_format}"
        data_uri = f"data:{mime};base64,{image_data}"

        models_to_try: list[str] = []
        if model:
            models_to_try.append(model)
        models_to_try.extend(m for m in _FALLBACK_MODELS if m and m != model)

        last_error = ""
        for m in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_PROMPT},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        },
                    ],
                    max_tokens=300,
                    temperature=0.3,
                )
                content = response.choices[0].message.content or ""
                if content:
                    return content
            except Exception as e:
                last_error = str(e)
                continue

        return f"[Erro: todos os modelos falharam — último erro: {last_error}]"
    except Exception as e:
        return f"[Erro: {e}]"


def describe_images_batch(
    images: list[dict],
    model: str | None = None,
    api_key: str | None = None,
    max_images: int = 10,
) -> list[dict]:
    if not images:
        return images

    described = []
    for i, img in enumerate(images):
        if i >= max_images:
            img["description"] = "[Pulado — limite de descrições atingido]"
            described.append(img)
            continue

        desc = describe_image(
            img.get("data", ""),
            image_format=img.get("format", "png"),
            model=model,
            api_key=api_key,
        )
        img["description"] = desc
        described.append(img)

    return described
