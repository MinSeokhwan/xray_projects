#include "PhysListEmPenelope.hh"

PhysListEmPenelope::PhysListEmPenelope(const G4String& name)
  :  G4VPhysicsConstructor(name)
{
    G4EmParameters* param = G4EmParameters::Instance();
    param->SetDefaults();
    param->SetMinEnergy(10*eV);
    param->SetMaxEnergy(10*TeV);
    param->SetNumberOfBinsPerDecade(10);
  
    param->SetVerbose(0);
    param->Dump();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListEmPenelope::~PhysListEmPenelope()
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysListEmPenelope::ConstructProcess()
{
  // Add standard EM Processes

  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() ){
    G4ParticleDefinition* particle = particleIterator->value();
    G4ProcessManager* pmanager = particle->GetProcessManager();    
    G4String particleName = particle->GetParticleName();

    //Applicability range for Penelope models
    //for higher energies, the Standard models are used   
    G4double highEnergyLimit = 1*GeV;
         
    if (particleName == "gamma") {
      // gamma         

      G4PhotoElectricEffect* phot = new G4PhotoElectricEffect();
      G4PenelopePhotoElectricModel* 
      photModel = new G4PenelopePhotoElectricModel();
      photModel->SetHighEnergyLimit(highEnergyLimit);
      phot->AddEmModel(0, photModel);
      pmanager->AddDiscreteProcess(phot);
     
      G4ComptonScattering* compt = new G4ComptonScattering();
      G4PenelopeComptonModel* 
      comptModel = new G4PenelopeComptonModel();
      comptModel->SetHighEnergyLimit(highEnergyLimit);
      compt->AddEmModel(0, comptModel);
      pmanager->AddDiscreteProcess(compt);
     
      G4GammaConversion* conv = new G4GammaConversion();
      G4PenelopeGammaConversionModel* 
      convModel = new G4PenelopeGammaConversionModel();
      convModel->SetHighEnergyLimit(highEnergyLimit);
      conv->AddEmModel(0, convModel);
      pmanager->AddDiscreteProcess(conv);
     
      G4RayleighScattering* rayl = new G4RayleighScattering();
      G4PenelopeRayleighModel* 
      raylModel = new G4PenelopeRayleighModel();
      raylModel->SetHighEnergyLimit(highEnergyLimit);
      rayl->AddEmModel(0, raylModel);
      pmanager->AddDiscreteProcess(rayl);

    } else if (particleName == "e-") {
      //electron

      G4eIonisation* eIoni = new G4eIonisation();
      G4PenelopeIonisationModel* 
      eIoniModel = new G4PenelopeIonisationModel();
      eIoniModel->SetHighEnergyLimit(highEnergyLimit); 
      eIoni->AddEmModel(0, eIoniModel, new G4UniversalFluctuation() );
      pmanager->AddProcess(eIoni,                   -1,-1, 1);
      
      G4eBremsstrahlung* eBrem = new G4eBremsstrahlung();
      G4PenelopeBremsstrahlungModel* 
      eBremModel = new G4PenelopeBremsstrahlungModel();
      eBremModel->SetHighEnergyLimit(highEnergyLimit);
      eBrem->AddEmModel(0, eBremModel);
      pmanager->AddProcess(eBrem,                   -1,-1, 2);
                  
    } else if (particleName == "e+") {
      //positron
      G4eIonisation* eIoni = new G4eIonisation();
      G4PenelopeIonisationModel* 
      eIoniModel = new G4PenelopeIonisationModel();
      eIoniModel->SetHighEnergyLimit(highEnergyLimit); 
      eIoni->AddEmModel(0, eIoniModel, new G4UniversalFluctuation() );
      pmanager->AddProcess(eIoni,                   -1,-1, 1);
      
      G4eBremsstrahlung* eBrem = new G4eBremsstrahlung();
      G4PenelopeBremsstrahlungModel* 
      eBremModel = new G4PenelopeBremsstrahlungModel();
      eBremModel->SetHighEnergyLimit(highEnergyLimit);
      eBrem->AddEmModel(0, eBremModel);
      pmanager->AddProcess(eBrem,                   -1,-1, 2);      

      G4eplusAnnihilation* eAnni = new G4eplusAnnihilation();
      G4PenelopeAnnihilationModel* 
      eAnniModel = new G4PenelopeAnnihilationModel();
      eAnniModel->SetHighEnergyLimit(highEnergyLimit); 
      eAnni->AddEmModel(0, eAnniModel);
      pmanager->AddProcess(eAnni,                    0,-1, 3);
            
    } else if( particleName == "mu+" || 
               particleName == "mu-"    ) {
      //muon  
      pmanager->AddProcess(new G4MuIonisation,      -1,-1, 1);
      pmanager->AddProcess(new G4MuBremsstrahlung,  -1,-1, 2);
      pmanager->AddProcess(new G4MuPairProduction,  -1,-1, 3);       
     
    } else if( particleName == "alpha" || particleName == "GenericIon" ) { 
      pmanager->AddProcess(new G4ionIonisation,     -1,-1, 1);

    } else if ((!particle->IsShortLived()) &&
               (particle->GetPDGCharge() != 0.0) && 
               (particle->GetParticleName() != "chargedgeantino")) {
      //all others charged particles except geantino
      pmanager->AddProcess(new G4hIonisation,       -1,-1, 1);
    }    
  }
  
  // Deexcitation
  //
  G4VAtomDeexcitation* deex = new G4UAtomicDeexcitation();
  G4LossTableManager::Instance()->SetAtomDeexcitation(deex);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

