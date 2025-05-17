import streamlit as st
import json
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import matplotlib.pyplot as plt
# Your Groq API Key
GROQ_API_KEY = st.secrets["groq_api_key"]# Replace this with your key
GROQ_MODEL = "llama3-70b-8192"

# Streamlit UI
st.set_page_config(page_title="⚡ Circuit Generator", layout="centered")
st.title("🔌 Electrical Circuit Generator")
st.markdown("Generate circuits and simulate voltage/current across components (R, C, L)")

# Inputs
circuit_name = st.text_input("Circuit Description", "RC low-pass filter")
vin = st.text_input("Vin (Input Voltage)", "5V")
iin = st.text_input("Iin (Input Current)", "10mA")
frequency = st.text_input("Frequency (Hz)", "50Hz")
power_supply = st.text_input("Power Supply Values", "+5V DC")

if st.button("Generate Circuit") and circuit_name:
    with st.spinner("Generating with Groq..."):
        try:
            system_msg = """
You are an expert in electrical circuit design. When asked to generate a circuit:
1. Return a JSON object with "components" and "diagram" fields.
2. Escape the ASCII diagram correctly so it's valid JSON (e.g., use \\n for newlines).
3. Example output:

{
  "components": {
    "R1": {"type": "Resistor", "value": "1kΩ"},
    "C1": {"type": "Capacitor", "value": "0.1µF"}
  },
  "diagram": "Vin ── R1 ──+── Vout\\n            |\\n           C1\\n            |\\n           GND"
}
Only output this JSON. Do not explain anything.
"""

            user_msg = (
                f"Design a circuit called '{circuit_name}' using:\n"
                f"- Vin: {vin}, Iin: {iin}, Frequency: {frequency}, Power Supply: {power_supply}\n"
                "Output JSON with 'components' and 'diagram'."
            )

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            }

            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            response_json = res.json()
            content = response_json["choices"][0]["message"]["content"].strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            result = json.loads(content)
            components = result.get("components", {})
            diagram = result.get("diagram", "Diagram missing.")

        except Exception as e:
            st.error(f"❌ Error parsing Groq response: {str(e)}")
            st.stop()

    # ✅ Display Components & Diagram
    st.subheader("🧩 Components")
    st.json(components)

    st.subheader("📐 ASCII Diagram")
    st.code(diagram, language="text")

    st.subheader("🖼️ Diagram Image")
    def ascii_to_image(ascii_text):
        font = ImageFont.load_default()
        lines = ascii_text.split("\\n")
        max_width = max([len(line) for line in lines])
        padding = 20
        line_height = 15
        width = padding * 2 + max_width * 1
        height = padding * 2 + len(lines) * line_height
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((padding, padding + i * line_height), line, font=font, fill="black")
        return img

    image = ascii_to_image(diagram)
    st.image(image, caption="Rendered ASCII", use_container_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    st.download_button("⬇️ Download Diagram", img_bytes.getvalue(), "diagram.png", "image/png")

    st.download_button("⬇️ Download Components", json.dumps(components, indent=4),
                       "components.json", "application/json")

    # ✅ Waveform Simulation
    st.subheader("📊 Voltage & Current Waveforms Per Component")

    try:
        freq = float(''.join(filter(lambda x: x.isdigit() or x == ".", frequency)))
        vin_peak = float(''.join(filter(lambda x: x.isdigit() or x == ".", vin)))
        t = np.linspace(0, 1 / freq * 4, 1000)
        vin_wave = vin_peak * np.sin(2 * np.pi * freq * t)

        for name, comp in components.items():
            ctype = comp["type"].lower()
            value = comp["value"]
            st.markdown(f"### 🔹 {name} - {comp['type']} ({value})")

            try:
                if "k" in value:
                    val = float(value.replace("k", "").replace("Ω", "").strip()) * 1e3
                elif "m" in value and "F" not in value:
                    val = float(value.replace("m", "").strip()) * 1e-3
                elif "µ" in value or "u" in value:
                    val = float(value.replace("µ", "").replace("u", "").replace("F", "").replace("H", "").strip()) * 1e-6
                elif "n" in value:
                    val = float(value.replace("n", "").strip()) * 1e-9
                else:
                    val = float(''.join(filter(lambda x: x.isdigit() or x == ".", value)))
            except:
                st.warning(f"⚠️ Could not parse value for {name}")
                continue

            fig, ax = plt.subplots()
            if "resistor" in ctype:
                r = val
                i = vin_wave / r
                v = i * r
            elif "capacitor" in ctype:
                c = val
                i = np.gradient(vin_wave, t[1] - t[0]) * c
                v = np.cumsum(i) * (t[1] - t[0]) / c
            elif "inductor" in ctype:
                l = val
                i = vin_wave / 1
                v = l * np.gradient(i, t[1] - t[0])
            else:
                st.info("🛠 Component type not supported for waveform.")
                continue

            ax.plot(t, v, label=f"Voltage across {name} (V)", color='blue')
            ax.plot(t, i, label=f"Current through {name} (A)", color='green')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            st.pyplot(fig)

            st.markdown(f"- **Peak Voltage**: {np.max(np.abs(v)):.3f} V")
            st.markdown(f"- **Peak Current**: {np.max(np.abs(i)):.6f} A")

    except Exception as e:
        st.error(f"❌ Simulation error: {e}")
