import React from 'react';
import { CssVarsProvider } from '@mui/joy/styles'
import Sheet from "@mui/joy/Sheet"
import Stack from "@mui/joy/Stack"
import Logo from './Components/Logo';
import CreationContainer from './Containers/CreationContainer';
import { IModel } from './modelManager';


function App() {
  const [currentStep, setCurrentStep] = React.useState(0)
  const [model, setModel] = React.useState<IModel>()
  return (
    <CssVarsProvider defaultMode="dark">
      <Sheet variant="outlined" sx={{ height: "100vh", width: "100vw", border: "none", 
      display: "flex", flexDirection: "column", justifyContent: "space-between", 
      alignItems: "center"}}>
        <Stack alignItems="center" justifyContent="center" gap={2} pt={5} width="100%" height="100%">
          <Logo/>
            <CreationContainer currentStep={currentStep} setCurrentStep={setCurrentStep}
            model={model} setModel={setModel}/>
        </Stack>
      </Sheet>
    </CssVarsProvider>
  )
}

export default App;
