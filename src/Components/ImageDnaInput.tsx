import React from "react"
import Input from "@mui/joy/Input"

export default function ImageDnaInput({value, setValue}: {value: string, setValue: Function}) {
    return (
        <Input type="file" value={value} onChange={e => setValue(e.target.value)}
        slotProps={{input: {accept: "image/jpg, image/jpeg"}}}
        sx={{width: "15rem", p: 1}}></Input>
    )
}