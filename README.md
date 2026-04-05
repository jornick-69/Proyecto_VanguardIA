# VanguardIA UCE

Proyecto organizado en módulos para captura de video, detección, tracking, validación de eventos y notificación por Telegram.

## Uso rápido

1. Crea un entorno virtual.
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Crea un archivo `.env`, copia `.env.example` a `.env` y llena tus valores.
4. Ejecuta:
   ```bash
   python main.py
   ```

## Proveedores de validación
- `VALIDATION_PROVIDER=ollama`
- `VALIDATION_PROVIDER=gemini`

## Notas
- No dejes tus API keys en el código.
- Regenera las keys y tokens que estaban en el archivo original.
