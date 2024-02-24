import React from "react"
import { IModel } from "../modelManager"
import Typography from "@mui/joy/Typography"
import Divider from "@mui/joy/Divider"
import NavButton from "../Components/NavButton"
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded'
import Stack from "@mui/joy/Stack"
import DnaInput, { DNA } from "./DnaInput"
import GenerationModeSwitch, { GenerationMode } from "../Components/GenerationModeSelect"

interface ISetParametersContainerProps {
    model?: IModel,
    returnToPrevStage: Function,
    dna1: DNA | string,
    setDna1: Function,
    dna2: DNA | string,
    setDna2: Function,
}

export default function SetParametersContainer({ model, returnToPrevStage,
    dna1, setDna1, dna2, setDna2 }: ISetParametersContainerProps) {
    const [isDna1Image, setIsDna1Image] = React.useState(false)
    const [isDna2Image, setIsDna2Image] = React.useState(false)
    const [generationMode, setGenerationMode] = React.useState<GenerationMode>(GenerationMode.Single)
    return (
        <Stack justifyContent="space-between" height="100%">
            <div>
                <Typography level="title-lg">Set Parameters:</Typography>
                <Divider sx={{ borderWidth: "20px", my: 1 }} />
                <Typography level="title-lg" my={1}>Selected model: {model?.name}</Typography>
                <GenerationModeSwitch mode={generationMode} setMode={setGenerationMode} />
            </div>
            <Stack direction="row" gap={1}>
                <DnaInput value={dna1} setValue={setDna1} isImage={isDna1Image} setIsImage={setIsDna1Image} />
                {generationMode === GenerationMode.Transition &&
                <DnaInput value={dna2} setValue={setDna2} isImage={isDna2Image} setIsImage={setIsDna2Image} />}
            </Stack>
            <div>
                <Divider sx={{ borderWidth: "20px", my: 1 }} />
                <NavButton text="Select another model.." icon={<ArrowBackRoundedIcon />} handleClick={returnToPrevStage} />
            </div>

        </Stack>
    )
}