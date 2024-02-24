import React from "react"
import Switch from "@mui/joy/Switch"
import Card from "@mui/joy/Card"
import Typography from "@mui/joy/Typography"
import Divider from "@mui/joy/Divider"
import NumericDnaInput from "../Components/NumericDnaInput"
import ImageDnaInput from "../Components/ImageDnaInput"

export type DNA = [number, number, number, number]

interface IDnaInputProps {
    value: DNA | string,
    setValue: Function,
    isImage: boolean,
    setIsImage: Function,
}


export default function DnaInput({value, setValue, isImage, setIsImage}: IDnaInputProps) {
    const setNumericValue = (val: number, index: number) => {
        setValue((prev: DNA) => {
            const newValue = [...prev]
            prev[index] = Math.min(Math.max(val, -1), 1)
            return newValue
        })
    }
    return (
        <Card sx={{height: "20rem", width: "15rem"}}>
            <Typography level="title-md">Input DNA Parameters:</Typography>
            <Divider/>
            <Typography component="label" endDecorator={<Switch checked={isImage} 
            onChange={e => setIsImage(e.target.checked)}/> }>
                Use image?
            </Typography>
            <Divider/>
            {isImage ?
            <ImageDnaInput value={typeof value === "string" ? value : ""} setValue={setValue} />:
            <NumericDnaInput value={typeof value === "string" ? [0,0,0,0] : value} setValue={setNumericValue}/>}
        </Card>
    )
}