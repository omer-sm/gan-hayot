import React from "react"
import Card from "@mui/joy/Card"
import Input from "@mui/joy/Input"
import Typography from "@mui/joy/Typography"
import Divider from "@mui/joy/Divider"
import FormLabel from '@mui/joy/FormLabel'
import FormControl from '@mui/joy/FormControl'

interface ITransitionParamsInputProps {
    framerate: number,
    setFramerate: Function,
    frameCount: number,
    setFrameCount: Function,
    isActive: boolean,
}

export default function TransitionParamsInput({ framerate, setFramerate, frameCount,
    setFrameCount, isActive }: ITransitionParamsInputProps) {
    return (
        <Card sx={{ height: "20rem", width: "13rem" }}>
            <Typography level="title-md" color={isActive ? undefined : "neutral"}>Transition Parameters:</Typography>
            <Divider />
            <FormControl disabled={!isActive}>
                <FormLabel>Frame rate (FPS):</FormLabel>
                <Input type="number" value={framerate} onChange={e => setFramerate(e.target.valueAsNumber)} />
            </FormControl>
            <FormControl disabled={!isActive}>
                <FormLabel>Frame count:</FormLabel>
                <Input type="number" value={frameCount} onChange={e => setFrameCount(e.target.valueAsNumber)} />
            </FormControl>
            <FormControl disabled={!isActive}>
            <FormLabel>Output length: {(frameCount/framerate).toFixed(1)} seconds</FormLabel>
            </FormControl>
            
        </Card>
    )
}