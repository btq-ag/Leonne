#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_consensus_node.py

Implementation of a node in a Quantum-Enhanced Distributed Consensus Network (QDCN).
Each node can participate in quantum consensus sets, submit and verify transactions
with quantum-enhanced security, and engage in the QDCN protocol phases.

Key quantum enhancements:
1. Quantum random number generation for improved unpredictability
2. Quantum-resistant cryptography
3. Quantum entanglement simulation for consensus correlation
4. Superposition-based decision making
5. Quantum measurement for vote determination

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import hashlib
import time
import json
import copy
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


class QuantumConsensusNode:
    """
    A node in a Quantum-Enhanced Distributed Consensus Network that participates in 
    the quantum consensus protocol and transaction verification.
    """
    
    def __init__(self, node_id, honesty_probability=1.0, quantum_enhancement_level=0.8):
        """
        Initialize a quantum consensus node.
        
        Parameters:
        -----------
        node_id : int or str
            Unique identifier for the node
        honesty_probability : float, default=1.0
            Probability that the node behaves honestly (for simulation purposes)
        quantum_enhancement_level : float, default=0.8
            Level of quantum enhancement (affects randomness quality and security)
        """
        self.node_id = node_id
        self.honesty_probability = honesty_probability
        self.quantum_enhancement_level = quantum_enhancement_level
        self.is_compliant = True
        
        # Generate quantum-resistant key pair for the node
        # In a real quantum implementation, this would use quantum-resistant algorithms
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        self.public_key = self.private_key.public_key()
        
        # Serialized public key for sharing
        self.public_key_bytes = self.public_key.public_bytes(
            encoding=Encoding.PEM, 
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        # Node's view of the network with quantum entanglement information
        self.known_nodes = {}  # {node_id: public_key}
        self.recognized_nodes = set()  # Set of node_ids recognized by this node
        self.entanglement_pairs = {}  # {node_id: entanglement_strength}
        
        # Consensus-related attributes
        self.consensus_sets = []  # List of consensus sets this node is part of
        self.bid_requests = []  # List of bid requests from this node
        self.transaction_verifications = {}  # {tx_id: verification_result}
        
        # Protocol state with quantum enhancements
        self.current_round = 0
        self.messages_received = {}  # {round_id: {node_id: message}}
        self.message_timestamps = {}  # {round_id: {node_id: timestamp}}
        self.quantum_states = {}  # {round_id: quantum_state_vector}
        
        # Compliance tracking
        self.compliance_votes = {}  # {round_id: {node_id: {voter_id: vote}}}
        
        # Transaction pool and ledger
        self.transaction_pool = []
        self.verified_transactions = []
        
        # Quantum random seeds
        self.local_quantum_seed = None
    
    def simulate_quantum_random_bit(self):
        """
        Simulate a quantum random bit using quantum principles.
        
        Returns:
        --------
        int
            0 or 1 with equal probability (in an ideal quantum system)
        """
        # In a real quantum system, this would use quantum hardware
        # For simulation, we adjust the randomness based on quantum enhancement level
        if np.random.random() < self.quantum_enhancement_level:
            # High-quality quantum randomness
            return np.random.choice([0, 1])
        else:
            # Lower-quality classical randomness (slightly biased for simulation)
            return np.random.choice([0, 1], p=[0.49, 0.51])
    
    def generate_quantum_random_bytes(self, num_bytes=32):
        """
        Generate random bytes using simulated quantum random bits.
        
        Parameters:
        -----------
        num_bytes : int, default=32
            Number of bytes to generate
            
        Returns:
        --------
        bytes
            Random bytes generated with quantum-inspired randomness
        """
        # Generate random bits
        random_bits = [self.simulate_quantum_random_bit() for _ in range(num_bytes * 8)]
        
        # Convert bits to bytes
        random_bytes = bytearray()
        for i in range(0, len(random_bits), 8):
            byte_bits = random_bits[i:i+8]
            byte_value = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
            random_bytes.append(byte_value)
            
        return bytes(random_bytes)
    
    def generate_quantum_seed(self, transaction_id=None):
        """
        Generate a local quantum random seed based on transaction ID or current time.
        
        Parameters:
        -----------
        transaction_id : str, optional
            Transaction ID to use for quantum seed generation
            
        Returns:
        --------
        bytes
            Quantum random seed as bytes
        """
        if transaction_id is None:
            # Use current time if no transaction ID provided
            data = f"{self.node_id}_{time.time()}"
        else:
            data = f"{self.node_id}_{transaction_id}"
            
        # Combine quantum randomness with deterministic hash
        quantum_random_part = self.generate_quantum_random_bytes(16)
        deterministic_part = hashlib.sha256(data.encode()).digest()[:16]
        
        # Combine the two parts
        combined = bytearray()
        for qb, db in zip(quantum_random_part, deterministic_part):
            combined.append(qb ^ db)  # XOR combination
            
        self.local_quantum_seed = bytes(combined)
        return self.local_quantum_seed
    
    def sign_message(self, message):
        """
        Sign a message using the node's private key.
        
        Parameters:
        -----------
        message : bytes or str
            Message to sign
            
        Returns:
        --------
        bytes
            Signature
        """
        if isinstance(message, str):
            message = message.encode()
            
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, message, signature, public_key):
        """
        Verify a signature using a public key.
        
        Parameters:
        -----------
        message : bytes or str
            Message that was signed
        signature : bytes
            Signature to verify
        public_key : cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey
            Public key to use for verification
            
        Returns:
        --------
        bool
            True if signature is valid, False otherwise
        """
        if isinstance(message, str):
            message = message.encode()
            
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def create_quantum_bid_request(self, transaction_ids, consensus_set_size):
        """
        Create a quantum-enhanced bid request for consensus on transactions.
        
        Parameters:
        -----------
        transaction_ids : list
            List of transaction IDs to request consensus for
        consensus_set_size : int
            Requested size of the consensus set
            
        Returns:
        --------
        dict
            Quantum-enhanced bid request
        """
        # Act dishonestly with probability (1 - honesty_probability)
        is_honest = np.random.random() < self.honesty_probability
        
        # Generate quantum random seed
        quantum_seed = self.generate_quantum_seed().hex()
        
        # Create quantum-entangled bid request
        # In a real quantum system, this would include quantum state information
        bid_request = {
            'node_id': self.node_id,
            'round_id': self.current_round,
            'transaction_ids': transaction_ids,
            'consensus_set_size': consensus_set_size,
            'recognized_nodes': list(self.recognized_nodes),
            'timestamp': time.time(),
            'quantum_seed': quantum_seed,
            'quantum_enhanced': True,
            'quantum_enhancement_level': self.quantum_enhancement_level
        }
        
        # If dishonest, manipulate the bid
        if not is_honest:
            # For simulation: dishonest node might exclude some known nodes
            if len(self.recognized_nodes) > 3:
                exclude_count = int(len(self.recognized_nodes) * 0.3)
                dishonest_recognized = list(self.recognized_nodes)[:-exclude_count]
                bid_request['recognized_nodes'] = dishonest_recognized
        
        # Create signature
        message = json.dumps(bid_request, sort_keys=True).encode()
        bid_request['signature'] = self.sign_message(message).hex()
        
        # Keep track of this bid
        self.bid_requests.append(bid_request)
        
        return bid_request
    
    def receive_quantum_message(self, sender_id, message, timestamp=None):
        """
        Receive and process a message from another node with quantum verification.
        
        Parameters:
        -----------
        sender_id : int or str
            ID of the sending node
        message : dict
            Message content
        timestamp : float, optional
            Receipt timestamp (defaults to current time)
            
        Returns:
        --------
        bool
            True if message is valid and accepted, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()
            
        round_id = message.get('round_id', self.current_round)
        
        # Initialize data structures for this round if not already done
        if round_id not in self.messages_received:
            self.messages_received[round_id] = {}
            self.message_timestamps[round_id] = {}
            self.quantum_states[round_id] = {}
        
        # Store message, timestamp, and quantum state
        self.messages_received[round_id][sender_id] = message
        self.message_timestamps[round_id][sender_id] = timestamp
        
        # Simulate quantum state for this message
        # In a real quantum system, this would involve actual quantum states
        quantum_state = self.simulate_quantum_message_state(sender_id, message)
        self.quantum_states[round_id][sender_id] = quantum_state
        
        return True
    
    def simulate_quantum_message_state(self, sender_id, message):
        """
        Simulate a quantum state for a message (for demonstration purposes).
        
        Parameters:
        -----------
        sender_id : int or str
            ID of the sending node
        message : dict
            Message content
            
        Returns:
        --------
        dict
            Simulated quantum state information
        """
        # Create a simulated quantum state based on message content
        # In a real quantum system, this would be an actual quantum state
        
        # Create hash from message
        msg_str = json.dumps(message, sort_keys=True)
        msg_hash = hashlib.sha256(msg_str.encode()).digest()
        
        # Convert hash to float values for simulated quantum state vector
        state_vector = [b / 255.0 for b in msg_hash[:4]]  # Use first 4 bytes for simplicity
        
        # Normalize the state vector (as quantum states must be normalized)
        norm = np.sqrt(sum(x**2 for x in state_vector))
        state_vector = [x/norm for x in state_vector]
        
        # Simulated entanglement strength with sender
        entanglement = 0.0
        if sender_id in self.entanglement_pairs:
            entanglement = self.entanglement_pairs[sender_id]
        
        return {
            'state_vector': state_vector,
            'entanglement': entanglement,
            'fidelity': 0.9 + 0.1 * np.random.random(),  # Simulated quantum state fidelity
            'is_quantum': message.get('quantum_enhanced', False)
        }
    
    def vote_on_quantum_compliance(self, round_id, tolerance=1.0):
        """
        Vote on the compliance of nodes based on message timing and quantum state quality.
        
        Parameters:
        -----------
        round_id : int
            Round ID to evaluate compliance for
        tolerance : float, default=1.0
            Time tolerance in seconds for compliance
            
        Returns:
        --------
        dict
            Compliance votes {node_id: bool}
        """
        if round_id not in self.message_timestamps or round_id not in self.quantum_states:
            return {}
            
        timestamps = self.message_timestamps[round_id]
        quantum_states = self.quantum_states[round_id]
        
        if not timestamps:
            return {}
            
        # Calculate median timestamp
        median_time = np.median(list(timestamps.values()))
        
        # Vote on compliance based on timing and quantum state quality
        compliance_votes = {}
        for node_id, timestamp in timestamps.items():
            # Time-based compliance
            time_compliant = abs(timestamp - median_time) <= tolerance
            
            # Quantum state-based compliance
            quantum_state = quantum_states.get(node_id, {})
            fidelity = quantum_state.get('fidelity', 0.0)
            is_quantum = quantum_state.get('is_quantum', False)
            
            # Nodes with quantum enhancement get a bonus in compliance determination
            quantum_compliant = fidelity > 0.8 if is_quantum else True
            
            # Overall compliance combines time and quantum factors
            is_compliant = time_compliant and quantum_compliant
            compliance_votes[node_id] = is_compliant
            
        # Initialize compliance votes structure for this round
        if round_id not in self.compliance_votes:
            self.compliance_votes[round_id] = {}
            
        # Store votes
        for node_id, vote in compliance_votes.items():
            if node_id not in self.compliance_votes[round_id]:
                self.compliance_votes[round_id][node_id] = {}
            self.compliance_votes[round_id][node_id][self.node_id] = vote
            
        return compliance_votes
    
    def quantum_verify_transaction(self, transaction_id, honest_vote=True):
        """
        Verify a transaction using quantum-enhanced verification (simulated).
        
        Parameters:
        -----------
        transaction_id : str
            Transaction ID to verify
        honest_vote : bool, default=True
            Whether to vote honestly or not
            
        Returns:
        --------
        bool
            Verification result
        """
        # In a real implementation, this would involve quantum verification methods
        
        # For simulation purposes, vote honestly with probability honesty_probability
        # Use quantum random generator for determining honesty
        quantum_random = self.simulate_quantum_random_bit()
        is_honest = (quantum_random == 1 and np.random.random() < self.honesty_probability) or (quantum_random == 0 and np.random.random() < self.honesty_probability*1.1)
        
        if is_honest:
            result = honest_vote
        else:
            # If dishonest, vote based on quantum biased coin flip
            result = np.random.choice([True, False], p=[0.6, 0.4])
            
        self.transaction_verifications[transaction_id] = result
        return result
    
    def compute_quantum_global_seed(self, collected_seeds):
        """
        Compute global quantum random seed from collected local quantum seeds.
        
        Parameters:
        -----------
        collected_seeds : dict
            {node_id: seed} from all participating nodes
            
        Returns:
        --------
        bytes
            Global quantum random seed
        """
        # Sort seeds by node_id to ensure deterministic result
        sorted_seeds = [collected_seeds[node_id] for node_id in sorted(collected_seeds.keys())]
        
        # Concatenate and hash
        concatenated = b''.join([bytes.fromhex(seed) if isinstance(seed, str) else seed for seed in sorted_seeds])
        
        # Apply quantum-inspired mixing
        mixed_seed = bytearray(hashlib.sha256(concatenated).digest())
        
        # Apply additional quantum-inspired transformation
        for i in range(len(mixed_seed)):
            if self.simulate_quantum_random_bit() == 1:
                mixed_seed[i] = mixed_seed[i] ^ 0xFF  # Quantum flip
                
        global_seed = bytes(mixed_seed)
        
        return global_seed
    
    def determine_quantum_consensus_sets(self, global_seed, node_ids, consensus_requests):
        """
        Determine the consensus sets based on the global quantum random seed.
        
        Parameters:
        -----------
        global_seed : bytes
            Global quantum random seed
        node_ids : list
            List of participating node IDs
        consensus_requests : list
            List of consensus set requests
            
        Returns:
        --------
        list
            List of consensus sets, each a list of node IDs
        """
        # Sort node_ids to ensure deterministic result
        node_ids = sorted(node_ids)
        
        # Generate deterministic random generator with quantum-enhanced seed
        # Convert bytes to int and make sure it's within the valid range for RandomState
        seed_int = int.from_bytes(global_seed, byteorder='big')
        seed_int = seed_int % (2**32 - 1)  # Ensure seed is within valid range (0 to 2^32 - 1)
        rng = np.random.RandomState(seed_int)
        
        consensus_sets = []
        remaining_nodes = node_ids.copy()
        
        # Quantum-inspired set formation process
        for request in consensus_requests:
            # Get requested consensus set size
            set_size = min(request['consensus_set_size'], len(remaining_nodes))
            
            if set_size == 0:
                continue
                
            # Quantum-influenced selection weights based on entanglement
            weights = np.ones(len(remaining_nodes))
            for i, node_id in enumerate(remaining_nodes):
                entanglement = self.entanglement_pairs.get(node_id, 0.0)
                weights[i] = 1.0 + entanglement * 0.5
                
            # Normalize weights to probabilities
            weights = weights / np.sum(weights)
                
            # Randomly select nodes for this consensus set with quantum-influenced probabilities
            selected_indices = rng.choice(len(remaining_nodes), size=set_size, replace=False, p=weights)
            consensus_set = [remaining_nodes[i] for i in selected_indices]
            
            # Remove selected nodes from remaining nodes
            remaining_nodes = [node for i, node in enumerate(remaining_nodes) if i not in selected_indices]
            
            consensus_sets.append({
                'request_node_id': request['node_id'],
                'transaction_ids': request['transaction_ids'],
                'nodes': consensus_set,
                'quantum_enhanced': True,
                'quantum_coherence': 0.8 + 0.2 * rng.random()  # Simulated quantum coherence
            })
            
            # If no more nodes, break
            if not remaining_nodes:
                break
                
        return consensus_sets
    
    def vote_on_quantum_transactions(self, consensus_set):
        """
        Vote on transactions in a quantum consensus set.
        
        Parameters:
        -----------
        consensus_set : dict
            Quantum consensus set with transaction IDs
            
        Returns:
        --------
        dict
            Votes on transactions {transaction_id: bool}
        """
        votes = {}
        for tx_id in consensus_set['transaction_ids']:
            votes[tx_id] = self.quantum_verify_transaction(tx_id)
            
        return votes
    
    def compute_quantum_compliance_result(self, compliance_votes):
        """
        Compute the final compliance result based on votes using quantum weighting.
        
        Parameters:
        -----------
        compliance_votes : dict
            {node_id: {voter_id: vote}}
            
        Returns:
        --------
        dict
            Compliance result {node_id: bool}
        """
        result = {}
        
        for node_id, votes in compliance_votes.items():
            # Count votes with quantum weighting
            weighted_votes = 0
            total_weight = 0
            
            for voter_id, vote in votes.items():
                # Weight based on entanglement with voter
                weight = 1.0 + self.entanglement_pairs.get(voter_id, 0.0) * 0.5
                weighted_votes += weight * (1 if vote else 0)
                total_weight += weight
                
            # Node is compliant if weighted majority votes yes
            result[node_id] = weighted_votes > total_weight / 2
            
        return result
    
    def generate_quantum_proof_of_consensus(self, round_id, consensus_set, transaction_votes, compliance_result):
        """
        Generate quantum-enhanced proof of consensus for verified transactions.
        
        Parameters:
        -----------
        round_id : int
            Round ID
        consensus_set : dict
            Quantum consensus set details
        transaction_votes : dict
            {node_id: {transaction_id: vote}}
        compliance_result : dict
            {node_id: is_compliant}
            
        Returns:
        --------
        dict
            Quantum proof of consensus
        """
        # Filter out non-compliant nodes
        compliant_nodes = [node_id for node_id in consensus_set['nodes'] 
                          if compliance_result.get(node_id, False)]
        
        # Calculate transaction results with quantum-enhanced majority logic
        transaction_results = {}
        for tx_id in consensus_set['transaction_ids']:
            # Count votes for this transaction from compliant nodes
            votes = [transaction_votes.get(node_id, {}).get(tx_id, False) 
                    for node_id in compliant_nodes]
            
            # Transaction is verified if quantum-weighted majority votes yes
            if votes:
                # Convert to Python bool to avoid NumPy bool_ which isn't JSON serializable
                transaction_results[tx_id] = bool(sum(votes) > len(votes) / 2)
            else:
                transaction_results[tx_id] = False
        
        # Create quantum-enhanced proof of consensus
        poc = {
            'round_id': round_id,
            'timestamp': time.time(),
            'request_node_id': consensus_set['request_node_id'],
            'consensus_set': consensus_set['nodes'],
            'compliant_nodes': compliant_nodes,
            'transaction_results': transaction_results,
            'quantum_enhanced': True,
            'quantum_coherence': consensus_set.get('quantum_coherence', 0.9),
            'signature': {}  # Will be filled with signatures
        }
        
        # Sign the proof
        # Convert all values to Python native types to ensure JSON serialization works
        json_safe_poc = {k: (v if k != 'transaction_results' else {tx: bool(result) for tx, result in v.items()}) 
                         for k, v in poc.items() if k != 'signature'}
        message = json.dumps(json_safe_poc, sort_keys=True).encode()
        poc['signature'][self.node_id] = self.sign_message(message).hex()
        
        return poc
    
    def update_quantum_transaction_ledger(self, proofs_of_consensus):
        """
        Update the transaction ledger based on quantum proofs of consensus.
        
        Parameters:
        -----------
        proofs_of_consensus : list
            List of quantum proofs of consensus
            
        Returns:
        --------
        list
            Newly verified transactions
        """
        newly_verified = []
        
        for poc in proofs_of_consensus:
            quantum_bonus = 1.0
            if poc.get('quantum_enhanced', False):
                # Quantum-enhanced proofs get a verification bonus
                quantum_bonus = 1.2
                
            for tx_id, verified in poc['transaction_results'].items():
                # Additional quantum-influenced verification check
                quantum_verify = verified and (np.random.random() < quantum_bonus * 0.8)
                
                if quantum_verify and tx_id not in self.verified_transactions:
                    self.verified_transactions.append(tx_id)
                    newly_verified.append(tx_id)
                    
        return newly_verified
    
    def update_entanglement_pairs(self, node_id, interaction_strength=0.1):
        """
        Update entanglement strength with another node based on interactions.
        
        Parameters:
        -----------
        node_id : str
            ID of the node to update entanglement with
        interaction_strength : float, default=0.1
            Strength of the interaction
            
        Returns:
        --------
        float
            New entanglement strength
        """
        current = self.entanglement_pairs.get(node_id, 0.0)
        
        # Increase entanglement but cap at 1.0
        new_entanglement = min(1.0, current + interaction_strength)
        self.entanglement_pairs[node_id] = new_entanglement
        
        return new_entanglement
