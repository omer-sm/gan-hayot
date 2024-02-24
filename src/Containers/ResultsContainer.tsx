import React from "react"
import Stack from "@mui/joy/Stack"
import Typography from "@mui/joy/Typography"
import NavButton from "../Components/NavButton"
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded'

interface IResultsContainerProps {
    goToFirstStage: Function,
    
}

export default function ResultsContainer({goToFirstStage, }: IResultsContainerProps) {
    return (
        <Stack height="100%" sx={{alignItems: "center", justifyContent: "center", gap: 2}}>
            <Typography level="h2">Generating results! They should open in a new window.</Typography>
            <Typography level="body-lg">Check the console for more info.</Typography>
            <NavButton handleClick={goToFirstStage} text="Generate again!" icon={<ArrowBackRoundedIcon/>}/>
        </Stack>
    )
}