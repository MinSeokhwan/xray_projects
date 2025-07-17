#include "generator.hh"

#include "G4GeneralParticleSource.hh"
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>

//NSPrimaryGenerator::NSPrimaryGenerator()
//{   
//    G4int n_particles = 1;
//    fParticleGun = new G4ParticleGun(n_particles);
//
//    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
//    G4ParticleDefinition *particle = particleTable->FindParticle("gamma");
//
//    G4ThreeVector pos(0.,0.,-1.*mm);
//    fParticleGun->SetParticlePosition(pos);
//
//    G4ThreeVector mom(0.,0.,1.);
//
//    fParticleGun->SetParticleMomentumDirection(mom);
//
//    fParticleGun->SetParticleEnergy(100.*keV);
//
//    fParticleGun->SetParticleDefinition(particle);
//}

NSPrimaryGenerator::NSPrimaryGenerator()
{
    fParticleGun = new G4GeneralParticleSource();
    //std::cerr << "[DEBUG] Using G4GeneralParticleSource\n";
	//G4cout << "[G4cout] /gps commands should be available\n" << G4endl;
}

NSPrimaryGenerator::~NSPrimaryGenerator()
{
    delete fParticleGun;
}

void NSPrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    fParticleGun->GeneratePrimaryVertex(anEvent);
}
