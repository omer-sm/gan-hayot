import React from "react"
import Input from "@mui/joy/Input"
import Stack from "@mui/joy/Stack"
import {DNA} from "../Containers/DnaInput"

interface INumericDnaInputProps {
    value: DNA,
    setValue: Function,
    
}

export default function NumericDnaInput({value, setValue, }: INumericDnaInputProps) {
    return (
        <Stack>
            <Input type="number" value={value[0]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 0)}></Input>
            <Input type="number" value={value[1]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 1)}></Input>
            <Input type="number" value={value[2]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 2)}></Input>
            <Input type="number" value={value[3]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 3)}></Input>
        </Stack>
    )
}