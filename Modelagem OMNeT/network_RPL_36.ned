import inet.common.scenario.ScenarioManager;
import inet.physicallayer.contract.packetlevel.IRadioMedium;
import inet.visualizer.contract.IIntegratedVisualizer;
import rpl.RplRouter;
import inet.networklayer.configurator.ipv6.Ipv6FlatNetworkConfigurator;
import inet.physicallayer.common.packetlevel.RadioMedium;

network RplNetwork36
{
    parameters:
        int numNodes = default(35);
        int numSinks = default(1);
        int numAttack = default(1);
          @display("bgb=276,252");       // Background size adjusted for the 36-node layout
    submodules:
        visualizer: <default("IntegratedCanvasVisualizer")> like IIntegratedVisualizer if hasVisualizer() {
            @display("p=1550,2450");
            networkRouteVisualizer.displayRoutes = true;
        }
        radioMedium: <> like IRadioMedium {
            @display("p=644.47797,197.118;is=s");
        }
        configurator: Ipv6FlatNetworkConfigurator {
            @display("p=644.47797,118.83;is=s");
        }

        sink[numSinks]: RplRouter {
            @display("i=misc/sensor;p=120.0,120;is=s"); // Sink placed near the center of the scenario
        }
        host[numNodes]: RplRouter {
            @display("i=misc/sensor2;p=83.712,606.912");
        }

    connections allowunconnected:
        // Wireless links are handled by the radio medium and the RPL routing process
}
