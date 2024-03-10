import React from "react"
import Input from "@mui/joy/Input"
import Typography from "@mui/joy/Typography"
import Stack from "@mui/joy/Stack"
import Textarea from "@mui/joy/Textarea"

export default function ImageDnaInput({ value, setValue, isActive }: { value: string, setValue: Function, isActive: boolean }) {
    return (
        <Stack gap={0.5}>
            <Typography>Enter image path:</Typography>
            <Input disabled={!isActive} variant="soft" value={value}
                onChange={e => setValue(e.target.value)} sx={{ width: "15rem", p: 1, maxHeight: "10rem" }}
                component={Textarea} />
        </Stack>
    )
}