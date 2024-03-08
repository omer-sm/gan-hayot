import React from "react"
import Input from "@mui/joy/Input"

export default function ImageDnaInput({value, setValue, isActive}: {value: string, setValue: Function, isActive: boolean}) {
    return (
        <Input disabled={!isActive} variant="soft" type="file" value={value} onChange={e => setValue(e.target.value)}
        slotProps={{input: {accept: "image/jpg, image/jpeg"}}}
        sx={{width: "15rem", p: 1}}></Input>
    )
}