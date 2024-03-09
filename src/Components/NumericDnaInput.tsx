import React from "react"
import Input from "@mui/joy/Input"
import Stack from "@mui/joy/Stack"
import {DNA} from "../Containers/DnaInput"

interface INumericDnaInputProps {
    value: DNA,
    setValue: Function,
    isActive: boolean
}

export default function NumericDnaInput({value, setValue, isActive}: INumericDnaInputProps) {
    return (
        <Stack gap={1}>
            <Input variant="soft" type="number" value={value[0]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 0)} disabled={!isActive}></Input>
            <Input variant="soft" type="number" value={value[1]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 1)} disabled={!isActive}></Input>
            <Input variant="soft" type="number" value={value[2]} slotProps={{input: {min: -1, max: 1, step: 0.1,}}}
            onChange={e => setValue(e.target.valueAsNumber, 2)} disabled={!isActive}></Input>
        </Stack>
    )
}