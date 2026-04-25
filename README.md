After the user clicks “Open Video”, the player will pass the local video file path to the algorithm:

video_path: Path

The algorithm should use this path to load and process the video, audio, frames, transcripts, etc.

The player will not pass preloaded frames or audio arrays, only the video file path.

We recommend that the algorithm returns a list[dict], where each dictionary represents a segment:

[

    {
        "start": 0.0,
        
        "end": 12.5,
        
        "label": "Intro"
        
    },
    
    {
        "start": 12.5,
        
        "end": 180.0,
        
        "label": "Core Content"
    },
    
    {
        "start": 180.0,
        
        "end": 205.0,
        
        "label": "Advertisement"
    }
    
]

The player will use build_segment_from_payload() to convert each payload dictionary into the internal Segment format required by the player.

Requirements:

start and end are in seconds and may be decimal values. The player converts them to milliseconds for playback seeking.

end must be greater than start

Segments should be ordered by time

Segments should not overlap

Segments should ideally cover the entire video (avoid gaps in the timeline)

label must match the predefined labels used in the player

Current Standard Labels:

Core Content

Intro

Outro

Advertisement

Self-Promotion

Recap

Transition

Inactivity

Filler

To see how segments are generated and passed to the player, start from SegmentationWorker. It runs run_video_segmentation(video_path) in the background and emits the resulting segments back to the UI.
