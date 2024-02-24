import {DNA} from "./Containers/DnaInput"

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

const getImageDna = async (path:string, model: IModel):Promise<DNA> => {
    return [0.5, 0.5, 0.5, 0.5]
}

export {getModels}