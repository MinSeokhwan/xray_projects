#include <iostream>

#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "QGSP_BERT.hh"
#include "Randomize.hh" //new addition
#include "time.h" //new addition

#include "construction.hh"
#include "physics.hh"
#include "action.hh"

void disable_stdout()
{
	FILE* tmp = fopen("/dev/null", "a+");

	fflush(stdout);

	auto outfd = dup(STDOUT_FILENO);

	if (outfd == -1 || dup2(fileno(tmp), STDOUT_FILENO) == -1)
		G4cout<<"failed to disable stdout"<<G4endl;

	fclose(tmp);
}

int main(int argc, char** argv)
{
    disable_stdout();
        G4UIExecutive* ui = 0;

    #ifdef G4MULTITHREADED
      G4MTRunManager* runManager = new G4MTRunManager;
    #else
      G4RunManager* runManager = new G4RunManager;
    #endif

    //choose the Random engine
    CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine()); //new addition
    //set random seed with system time
    G4long seed = time(NULL); //new addition
    CLHEP::HepRandom::setTheSeed(seed); //new addition

    runManager->SetUserInitialization(new NSDetectorConstruction());
    runManager->SetUserInitialization(new NSPhysicsList());
    runManager->SetUserInitialization(new NSActionInitialization());

    // G4VModularPhysicsList* physics = new QGSP_BERT();
    // runManager->SetUserInitialization(physics);

    if (argc == 1)
    {
        ui = new G4UIExecutive(argc, argv);
    }

    G4VisManager *visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager *UImanager = G4UImanager::GetUIpointer();

    if(ui)
    {
        UImanager->ApplyCommand("/control/execute vis.mac");
        ui->SessionStart();
    }
    else
    {
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command+fileName);

        if (argc == 4)
        {
          int nEvents = std::atoi(argv[2]);
          G4String tau_str = argv[3];
          G4String cmd = "mkdir " + tau_str;
          system(cmd);
          for (int i = 0; i < nEvents; ++i)
          {
            runManager->BeamOn(1);
            G4String output_file_name = "mv output0.root ./" + tau_str + "/output" + std::to_string(i+1) + ".root";
            system(output_file_name);
          }
        } 
    }

    //job termination
    delete visManager;
    delete runManager;
    return 0;
}
