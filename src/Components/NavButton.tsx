import React from "react"
import Button from "@mui/joy/Button"
import Typography from "@mui/joy/Typography"

interface INavButtonProps {
    text: string,
    icon: React.ReactNode,
    handleClick: Function,
    
}

export default function NavButton({text, icon, handleClick, }: INavButtonProps) {
    return (
            <Button onClick={() => handleClick()}
             sx={{width: "15rem", height: "4rem", justifyContent: "start", pr: 6, gap: 1}} 
             variant="soft" color="primary" startDecorator={icon}>
                <Typography level="body-md" fontSize={18} textAlign="start">{text}</Typography>
            </Button>
    )
}