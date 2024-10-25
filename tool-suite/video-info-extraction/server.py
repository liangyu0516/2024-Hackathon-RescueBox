import json
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
    response_json = dict()
    for file, qs in analysis_results.items():
        response_json[file] = {}
        for q, rs in qs.items():
            response_json[file][q] = [
                {
                    "info": r["info"],
                    "start_time": f"{r['start_time']:05.2f}",
                    "end_time": f"{r['end_time']:05.2f}"
                }
                for r in rs
            ]

    # Return the JSON-structured response
    return ResponseBody(
        root=(
            BatchTextResponse(
                texts=[
                    TextResponse(
                        title=file,
                        value=json.dumps(response_json, indent=3)  # Convert each file's response to JSON string format
                    )
                    for file in response_json.keys()
                ]
            )
        )
    )


if __name__ == "__main__":
    server.run()
