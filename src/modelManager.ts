import {DNA} from "./Containers/DnaInput"
import { GenerationMode } from "./Components/GenerationModeSelect"

export interface IModel {
    name: string,
    path: string,
    emoji: string,
}

const getModels = ():IModel[] => {
    return [{name: "Cats", path: "", emoji: "🐱"},
    {name: "Horses", path: "", emoji: "🐴"},
    {name: "TurtleMonkey", path: "", emoji: "🐢"}]
}

const makeGenCommand = (model:IModel, mode:GenerationMode, dna1:DNA|string,
     dna2:DNA|string|null, framerate:number, frameCount:number):string => {
    return "?GEN|" + (mode === GenerationMode.Single ? "IMG|" : "GIF|") + 
    model.path + "|" + framerate + "|" + frameCount + "|" + dna1 + "|" + dna2 + "?"
}

export {getModels, makeGenCommand}