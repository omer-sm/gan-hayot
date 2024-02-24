import React from "react"
import Stack from "@mui/joy/Stack"
import { getModels, IModel } from "../modelManager"
import ModelListItem from "../Components/ModelListItem"
import Typography from "@mui/joy/Typography"
import Divider from "@mui/joy/Divider"


const makeModelsList = (lst: IModel[], selectModel: Function) => {
    return lst.map(model => <ModelListItem model={model} selectModel={selectModel} />)
}

export default function ChooseModelContainer({ selectModel }: { selectModel: Function }) {

    return (
        <>
            <Typography level="title-lg">Choose Model:</Typography>
            <Divider sx={{ borderWidth: "20px" }} />
            <Stack gap={1} sx={{ py: 1, px: 0, borderRadius: "5px", height: "90%" }}>
                {makeModelsList(getModels(), selectModel)}
            </Stack>
        </>
    )
}