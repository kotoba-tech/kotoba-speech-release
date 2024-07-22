import io
import json
import os

import gradio as gr
import requests
import soundfile as sf

API_SERVER_URL = "http://127.0.0.1:58003/tts"
RADIO_CHOICES = ["Preset voices", "Upload target voice", "Record your voice"]
MAX_CHARS = 220
PRESET_VOICES = {
    # female
    "Ava": "https://cdn.themetavoice.xyz/speakers/ava.flac",
    "Bria": "https://cdn.themetavoice.xyz/speakers/bria.mp3",
    # male
    "Alex": "https://cdn.themetavoice.xyz/speakers/alex.mp3",
    "Jacob": "https://cdn.themetavoice.xyz/speakers/jacob.wav",
}


def denormalise_top_p(top_p):
    # returns top_p in the range [0.9, 1.0]
    return round(0.9 + top_p / 100, 2)


def denormalise_guidance(guidance):
    # returns guidance in the range [1.0, 3.0]
    return 1 + ((guidance - 1) * (3 - 1)) / (5 - 1)


def _handle_edge_cases(to_say, upload_target):
    if not to_say:
        raise gr.Error("Please provide text to synthesise")

    def _check_file_size(path):
        if not path:
            return
        filesize = os.path.getsize(path)
        filesize_mb = filesize / 1024 / 1024
        if filesize_mb >= 50:
            raise gr.Error(
                f"Please upload a sample less than 20MB for voice cloning. Provided: {round(filesize_mb)} MB"
            )

    _check_file_size(upload_target)


def tts(to_say, top_p, guidance, toggle, preset_dropdown, upload_target, record_target):
    d_top_p = denormalise_top_p(top_p)
    d_guidance = denormalise_guidance(guidance)

    _handle_edge_cases(to_say, upload_target)

    to_say = to_say if len(to_say) < MAX_CHARS else to_say[:MAX_CHARS]

    custom_target_path = None
    if toggle == RADIO_CHOICES[1]:
        custom_target_path = upload_target
    elif toggle == RADIO_CHOICES[2]:
        custom_target_path = record_target

    config = {
        "text": to_say,
        "guidance": d_guidance,
        "top_p": d_top_p,
        "speaker_ref_path": PRESET_VOICES[preset_dropdown] if toggle == RADIO_CHOICES[0] else None,
    }
    headers = {"Content-Type": "audio/wav", "X-Payload": json.dumps(config)}
    if not custom_target_path:
        response = requests.post(API_SERVER_URL, headers=headers, data=None)
    else:
        with open(custom_target_path, "rb") as f:
            data = f.read()
            response = requests.post(API_SERVER_URL, headers=headers, data=data)

    wav, sr = None, None
    if response.status_code == 200:
        audio_buffer = io.BytesIO(response.content)
        audio_buffer.seek(0)
        wav, sr = sf.read(audio_buffer, dtype="float32")
    else:
        print(f"Something went wrong. response status code: {response.status_code}")

    return sr, wav


def change_voice_selection_layout(choice):
    index = RADIO_CHOICES.index(choice)
    return [
        gr.update(visible=True)
        if i == index else gr.update(visible=False)
        for i in range(len(RADIO_CHOICES))
    ]


title = "# TTS by Kotoba-Speech"

description = """
<strong>Kotoba-Speech v0.1</strong>は、1.2Bのトランスフォーマーに基づく音声生成モデルです。
以下の機能をサポートしています：
\n
* 日本語における滑らかなテキスト読み上げ生成
* スピーチプロンプトを通じたOne-shot音声クローニング

Kotoba Technologiesは、公開されたモデルを商用可能なApache 2.0ライセンスで公開します。
推論およびモデルコードは、Meta-Voiceをベースに作られており、学習コードは弊社のGitHubで近日中に公開する予定です。  
Kotoba Technologiesは、音声基盤モデルの開発に取り組んでおり、今後もモデルの公開を行なっていきます。是非、[Discord Community](https://discord.gg/qPVFqhGN7Z)に参加してご意見ください！

<strong>Kotoba-Speech v0.1</strong> is a 1.2B Transformer-based speech generative model. It supports the following properties:
\n
* Fluent text-to-speech generation in Japanese
* One-shot voice cloning through speech prompt

We are releasing our model under the Apache 2.0 license. Our inference and model code is adapted from Meta-Voice, and we will release our training code on our GitHub repository shortly.  
Kotoba Technologies is committing on developing speech foundation models, and we’ll continue releasing our models. Please join [our discord](https://discord.gg/qPVFqhGN7Z) to contribute to our community.
"""

with gr.Blocks(title="TTS by Kotoba-Speech") as demo:
    gr.Markdown(title)

    with gr.Row():
        gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            to_say = gr.TextArea(
                label="What should I say!?",
                lines=4,
                value="コトバテクノロジーズのミッションは、音声基盤モデルを作ることです。",
            )

            with gr.Row(), gr.Column():
                # voice settings
                top_p = gr.Slider(
                    value=5.0,
                    minimum=0.0,
                    maximum=10.0,
                    step=1.0,
                    label="Speech Stability - improves text following for a challenging speaker",
                )
                guidance = gr.Slider(
                    value=5.0,
                    minimum=1.0,
                    maximum=5.0,
                    step=1.0,
                    label="Speaker similarity - How closely to match speaker identity and speech style.",
                )

                # voice select
                toggle = gr.Radio(choices=RADIO_CHOICES, label="Choose voice", value=RADIO_CHOICES[0])

            with gr.Row(visible=True) as row_1:
                preset_dropdown = gr.Dropdown(
                    PRESET_VOICES.keys(), label="Preset voices", value=list(PRESET_VOICES.keys())[0]
                )
                with gr.Accordion("Preview: Preset voices", open=False):
                    for label, path in PRESET_VOICES.items():
                        gr.Audio(value=path, label=label)

            with gr.Row(visible=False) as row_2:
                upload_target = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload a clean sample to clone. Sample should contain 1 speaker, be between 10-90 seconds and not contain background noise.",
                    min_length=10,
                    max_length=90,
                )

            with gr.Row(visible=False) as row_3:
                record_target = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record your voice with a microphone to clone. Sample should contain 1 speaker, be between 10-90 seconds and not contain background noise.",
                    min_length=10,
                    max_length=90,
                )

            toggle.change(
                change_voice_selection_layout,
                inputs=toggle,
                outputs=[row_1, row_2, row_3],
            )

        with gr.Column():
            speech = gr.Audio(
                type="numpy",
                label="Kotoba-Speech says...",
            )

    submit = gr.Button("Generate Speech")
    submit.click(
        fn=tts,
        inputs=[to_say, top_p, guidance, toggle, preset_dropdown, upload_target, record_target],
        outputs=speech,
    )


demo.queue(default_concurrency_limit=2)
demo.launch()
# demo.launch(server_name="0.0.0.0", server_port=3000, share=True)  # dev
