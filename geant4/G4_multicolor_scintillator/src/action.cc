#include "action.hh"

NSActionInitialization::NSActionInitialization()
{}

NSActionInitialization::~NSActionInitialization()
{}

void NSActionInitialization::BuildForMaster() const
{
    NSRunAction *runAction = new NSRunAction();
    SetUserAction(runAction);
}

void NSActionInitialization::Build() const
{
    NSPrimaryGenerator *generator = new NSPrimaryGenerator();
    SetUserAction(generator);
    
    NSRunAction *runAction = new NSRunAction();
    SetUserAction(runAction);
    
    NSEventAction *eventAction = new NSEventAction(runAction);
    SetUserAction(eventAction);
    
    NSSteppingAction *steppingAction = new NSSteppingAction(eventAction);
    SetUserAction(steppingAction);
}