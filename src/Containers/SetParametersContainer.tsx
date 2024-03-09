import React from "react"
import { IModel } from "../modelManager"
import Typography from "@mui/joy/Typography"
import Divider from "@mui/joy/Divider"
import NavButton from "../Components/NavButton"
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded'
import Stack from "@mui/joy/Stack"
import DnaInput, { DNA } from "./DnaInput"
import GenerationModeSwitch, { GenerationMode } from "../Components/GenerationModeSelect"
import DrawRoundedIcon from '@mui/icons-material/DrawRounded'
import TransitionParamsInput from "../Components/TransitionParamsInput"

interface ISetParametersContainerProps {
    model?: IModel,
    returnToPrevStage: Function,
    dna1: DNA | string,
    setDna1: Function,
    dna2: DNA | string,
    setDna2: Function,
    generationMode: GenerationMode,
    setGenerationMode: Function,
    generate: Function,
    framerate: number,
    setFramerate: Function,
    frameCount: number,
    setFrameCount: Function,
}

export default function SetParametersContainer({ model, returnToPrevStage,
    dna1, setDna1, dna2, setDna2, generationMode, setGenerationMode, generate, framerate,
    setFramerate, frameCount, setFrameCount }: ISetParametersContainerProps) {
    const [isDna1Image, setIsDna1Image] = React.useState(false)
    const [isDna2Image, setIsDna2Image] = React.useState(false)
    return (
        <Stack justifyContent="space-between" height="100%">
            <div>
                <Typography level="title-lg">Set Parameters:</Typography>
                <Divider sx={{ borderWidth: "20px", my: 1 }} />
                <Typography level="title-lg" my={1}>Selected model: {model?.name}</Typography>
                <GenerationModeSwitch mode={generationMode} setMode={setGenerationMode} />
            </div>
            <Stack direction="row" gap={1} justifyContent="space-between">
                <DnaInput isActive={true} value={dna1} setValue={setDna1} isImage={isDna1Image} setIsImage={setIsDna1Image} />
                    <DnaInput isActive={generationMode === GenerationMode.Transition} value={dna2} setValue={setDna2} isImage={isDna2Image} setIsImage={setIsDna2Image} />
                <Stack gap={1}>
                    <TransitionParamsInput frameCount={frameCount} framerate={framerate} 
                    isActive={generationMode === GenerationMode.Transition} setFrameCount={setFrameCount}
                    setFramerate={setFramerate}/>
                    <NavButton text="Generate!" icon={<DrawRoundedIcon />} handleClick={generate} />
                </Stack>
            </Stack>
            <div>
                <Divider sx={{ borderWidth: "20px", my: 1 }} />
                <NavButton text="Select another model.." icon={<ArrowBackRoundedIcon />} handleClick={returnToPrevStage} />
            </div>

        </Stack>
    )
}