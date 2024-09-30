### Quantum GAN

#A Hybrid Quantum-Classical Approach to Synthetic Data Generation

# Tabular Qauntum Generator 
import pennylane as qml

@qml.qnode(dev, diff_method="parameter-shift")
def quanutm_circuit(noise, weights):
    weights = weights.reshape(q_depth, n_qubits)
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
    
    for i in range(q_depth):
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y+1])
    return qml.probs(wires=list(range(n_qubits)))

def partial_measure(noise, weights):
    probs = quanutm_circuit(noise, weights)
    probsgiven0  = probs[: (2** (n_qubits - n_a_qubits))]
    probsgiven0 /= tf.reduce_sum(probs)

    return probsgiven0 / tf.reduce_max(probsgiven0)


class PatchQuantumGenerator(nn.Module):

    def __init__(self, n_generators, output_dim, q_delta=1):
        super().__init__()

        self.q_params = nn.ParameterList(
            [nn.parameter(q_delta *torch.rand(q_dpeth*n_qubits), requires_grad=True) for _ in range(n_generators)]
        )
        self.n_generators = n_generators
        self. output_dim = output_dim

    def forward(self, x):
        patch_size = 2 ** (n_qubits - n_a_qubits)
        total_patches = (self.output_dim + patch_size - 1) // patch_size
        fake = torch.Tensor(x.size(0), 0).to(deice)

        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(device)

            for elem in x:
                q_out = partial_measure(elem, params).float().unsqeeze(0)
                patches = torch.cat((patches, q_out))
            
            fake = torch.cat((fake, patches), 1)
        
        fake = fake[:, :self.output_dim]
        return fake 
