import React from "react"
import Sheet from "@mui/joy/Sheet"
import Stack from "@mui/joy/Stack"
import Stepper from "@mui/joy/Stepper"
import Step from "@mui/joy/Step"
import StepIndicator from "@mui/joy/StepIndicator"
import ChooseModelContainer from "./ChooseModelContainer"
import { IModel } from "../modelManager"
import Card from "@mui/joy/Card"
import SetParametersContainer from "./SetParametersContainer"
import {DNA} from './DnaInput';
import { makeGenCommand } from "../modelManager"
import { GenerationMode } from "../Components/GenerationModeSelect"
import ResultsContainer from "./ResultsContainer"

const { ipcRenderer } = window.require('electron');

interface ICreationContainerProps {
    currentStep: number,
    setCurrentStep: Function,
    model?: IModel,
    setModel: Function
}
export default function CreationContainer({ currentStep, setCurrentStep, model, setModel }: ICreationContainerProps) {
    const selectModel = (selectedModel: IModel) => {
        setModel(selectedModel)
        setCurrentStep(1)
    }
    const returnToPrevStage = () => {
        if (currentStep > 0){
            setCurrentStep((x: number) => x-1)
        }
    }
    const returnToFirstStage = () => {
        setCurrentStep(0)
    }
    const [dna1, setDna1] = React.useState<DNA | string>([0,0,0,0])
    const [dna2, setDna2] = React.useState<DNA | string>([0,0,0,0])
    const [generationMode, setGenerationMode] = React.useState<GenerationMode>(GenerationMode.Single)
    const generate = () => {
        if (!model){
            return
        }
        setCurrentStep(2)
        ipcRenderer.send('call-py', makeGenCommand(model, generationMode, dna1, dna2));
    }
    return (
        <Sheet variant="outlined"
            sx={{
                height: "100%", width: "80%", borderWidth: "0.1em",
                borderRadius: "5px 5px 0 0", borderBottom: 0, bgcolor: "var(--joy-palette-neutral-800)",
                overflowY: "scroll", "::-webkit-scrollbar": {display: "none"}
            }}>
            <Stack alignItems="center" gap={2} pt={3} width="100%" height="100%">
                <Stepper sx={{ width: "70%" }}>
                    <Step indicator={
                        <StepIndicator variant={currentStep === 0 ? "solid" : "soft"}
                            color="primary">1</StepIndicator>
                    }>Choose Model</Step>
                    <Step indicator={
                        <StepIndicator variant={currentStep === 1 ? "solid" : "soft"}
                            color="primary">2</StepIndicator>
                    }>Set Parameters</Step>
                    <Step indicator={
                        <StepIndicator variant={currentStep === 2 ? "solid" : "soft"}
                            color="primary">3</StepIndicator>
                    }>Results</Step>
                </Stepper>
                <Card variant="outlined" sx={{ width: "70%", height: "85%" }}>
                    {[<ChooseModelContainer selectModel={selectModel} />,
                    <SetParametersContainer model={model} returnToPrevStage={returnToPrevStage}
                    dna1={dna1} setDna1={setDna1} dna2={dna2} setDna2={setDna2}
                    generationMode={generationMode} setGenerationMode={setGenerationMode}
                    generate={generate}/>,
                    <ResultsContainer goToFirstStage={returnToFirstStage}/>
                    ]
                    [currentStep]}
                </Card>
            </Stack>
        </Sheet>
    )
}