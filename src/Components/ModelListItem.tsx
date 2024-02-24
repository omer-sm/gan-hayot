import React from "react"
import ListItem from "@mui/joy/ListItem"
import ListItemDecorator from "@mui/joy/ListItemDecorator"
import Button from "@mui/joy/Button"
import { IModel } from "../modelManager"
import Typography from "@mui/joy/Typography"

interface IModelListItemProps {
    model: IModel,
    selectModel: Function,
}

export default function ModelListItem({ model, selectModel, }: IModelListItemProps) {
    return (
        <ListItem sx={{height: "3rem"}}>
            <Button onClick={() => selectModel(model)}
             sx={{width: "100%", height: "100%", justifyContent: "start", pr: 6, gap: 1}} 
             variant="soft" color="neutral">
                <ListItemDecorator sx={{fontSize: "1.3rem"}}>{model.emoji}</ListItemDecorator>
                <Typography level="title-lg" fontSize={22}>{model.name}</Typography>
            </Button>
        </ListItem>
    )
}