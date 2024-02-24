import React from "react"
import Typography from "@mui/joy/Typography"
import ToggleButtonGroup from "@mui/joy/ToggleButtonGroup"
import Button from "@mui/joy/Button"

export enum GenerationMode {
    Single = "SINGLE",
    Transition = "TRANSITION",
}

interface IGenerationModeSelectProps {
    mode: GenerationMode,
    setMode: Function,
}

export default function GenerationModeSelect({mode, setMode, }: IGenerationModeSelectProps) {
    return (
        <ToggleButtonGroup value={mode} onChange={(e, val) => setMode(val)}>
            <Button value={GenerationMode.Single}>Single</Button>
            <Button value={GenerationMode.Transition}>Transition</Button>
        </ToggleButtonGroup>
    )
}