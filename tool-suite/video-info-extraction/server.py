from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *

from .model import AudioTranscriptionModel, TextThreatDetectionModel

server = MLServer(__name__)


class TranscriptionInputs(TypedDict):
    audio_files: BatchFileInput


class NoParameters(TypedDict):
    pass


def task_schema_func():
    return TaskSchema(
        inputs=[
            InputSchema(
                key="audio_files",
                label="Audio Files",
                subtitle="Select the audio files to transcribe",
                input_type=InputType.BATCHFILE,
            )
        ],
        parameters=[],
    )


@server.route("/transcribe", task_schema_func=task_schema_func)
def transcribe(inputs: TranscriptionInputs, parameters: NoParameters) -> ResponseBody:
    print("Inputs:", inputs)
    print()
    print("Parameters:", parameters)
    print()
    files = [e.path for e in inputs['audio_files'].files]
    
    model = AudioTranscriptionModel()
    results = model.transcribe_batch(files)
    
    results = {r["file_path"]: r["result"] for r in results}

    threat_detection_model = TextThreatDetectionModel()
    analysis_results = {}
    for file_path, sentences in results.items():
        threat_results = dict()
        for sentence in sentences:
            detected_threats = threat_detection_model.detect_threats(sentence["text"])

            # Group results by question and filter by score
            for q, rs in detected_threats.items():
                if detected_threats[q]['score'] > 0.7:  # Filter based on score
                    threat_results[q] = threat_results.get(q, [])
                    threat_results[q].append({
                        'info': detected_threats[q]['answer'],
                        'start_time':sentence["start"],
                        'end_time': sentence["end"]
                    })
        # Store results in the analysis_results dictionary
        analysis_results[file_path] = threat_results
    
    # Format the output value
    response_text = dict()
    for file, qs in analysis_results.items():
        value = ""
        for q, rs in qs.items():
            value += q + "\n"
            value += "start end   information\n"
            for r in rs:
                value += f'{r["start_time"]:05.2f} {r["end_time"]:05.2f} {r["info"]}\n'
            value += "\n"
        response_text[file] = value

    return ResponseBody(
        root=(
            BatchTextResponse(
                texts=[
                    TextResponse(
                        title=file,
                        value=text
                    )
                    for file, text in response_text.items()
                ]
            )
        )
    )


if __name__ == "__main__":
    server.run()
