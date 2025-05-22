#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_consensus_network.py

Implementation of a Quantum-Enhanced Distributed Consensus Network (QDCN) that manages 
multiple quantum nodes, facilitates quantum-secure communication between them, 
and oversees the quantum consensus protocol.

Key quantum enhancements:
1. Quantum-inspired network topology with entanglement weighting
2. Quantum random number generation for protocol parameters
3. Enhanced security through quantum principles
4. Improved scalability using quantum-inspired consensus
5. Quantum-resilient validation mechanisms

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import time
import json
import networkx as nx
from quantum_consensus_node import QuantumConsensusNode


class QuantumConsensusNetwork:
    """
    A network of quantum consensus nodes that collectively form a Quantum-Enhanced 
    Distributed Consensus Network. Manages quantum-secure node communication, 
    quantum consensus set formation, and quantum protocol execution.
    """
    
    def __init__(self, num_nodes=10, honesty_distribution=None, 
                 network_connectivity=0.8, quantum_enhancement_level=0.8):
        """
        Initialize a quantum consensus network.
        
        Parameters:
        -----------
        num_nodes : int, default=10
            Number of nodes in the network
        honesty_distribution : list or None, default=None
            List of honesty probabilities for each node (if None, all nodes are honest)
        network_connectivity : float, default=0.8
            Probability of connection between any two nodes
        quantum_enhancement_level : float, default=0.8
            Level of quantum enhancement across the network
        """
        self.num_nodes = num_nodes
        self.quantum_enhancement_level = quantum_enhancement_level
        
        # Set honesty distribution
        if honesty_distribution is None:
            self.honesty_distribution = [1.0] * num_nodes
        else:
            if len(honesty_distribution) != num_nodes:
                raise ValueError("honesty_distribution must have length equal to num_nodes")
            self.honesty_distribution = honesty_distribution
            
        # Create quantum nodes
        self.nodes = {}
        for i in range(num_nodes):
            node_id = f"node_{i}"
            node_honesty = self.honesty_distribution[i]
            
            # Vary quantum enhancement slightly per node
            node_quantum_level = quantum_enhancement_level * (0.9 + 0.2 * np.random.random())
            
            # Create quantum node
            self.nodes[node_id] = QuantumConsensusNode(
                node_id=node_id, 
                honesty_probability=node_honesty,
                quantum_enhancement_level=node_quantum_level
            )
            
        # Initialize network graph
        self.network_graph = nx.Graph()
        for node_id in self.nodes:
            self.network_graph.add_node(node_id)
          # Initialize quantum entanglement data matrix (square matrix of size num_nodes x num_nodes)
        # This stores the entanglement strength between pairs of nodes
        self.entanglement_matrix = np.zeros((num_nodes, num_nodes))
            
        # Create quantum-weighted connections between nodes
        self._initialize_quantum_network_topology(network_connectivity)
        
        # Register nodes with each other
        self._register_quantum_nodes()
        
        # Initialize transaction data
        self.transaction_pool = {}  # {tx_id: transaction_data}
        self.transaction_ledger = {}  # {tx_id: verified_transaction_data}
        self.pending_transactions = []  # List of tx_ids pending verification
        
        # Quantum consensus protocol data
        self.current_round = 0
        self.consensus_sets_history = {}  # {round_id: consensus_sets}
        self.compliance_history = {}  # {round_id: compliance_results}
        self.transaction_votes_history = {}  # {round_id: {node_id: {tx_id: vote}}}
    
    def _initialize_quantum_network_topology(self, connectivity):
        """
        Initialize the quantum network topology with entanglement-weighted connections.
        
        Parameters:
        -----------
        connectivity : float
            Base connectivity probability
        """
        # Generate quantum-inspired connection probabilities
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                node_i = f"node_{i}"
                node_j = f"node_{j}"
                
                # Base probability adjusted by quantum effects
                quantum_factor = np.abs(np.sin(i*j*np.pi/self.num_nodes))
                connect_prob = connectivity * (0.8 + 0.4 * quantum_factor)
                
                # Add edge with probability
                if np.random.random() < connect_prob:
                    # Edge weight represents entanglement quality
                    entanglement = 0.5 + 0.5 * np.random.random() * self.quantum_enhancement_level
                    self.network_graph.add_edge(node_i, node_j, weight=entanglement)
                    
                    # Store initial entanglement
                    i_idx = int(node_i.split('_')[1])
                    j_idx = int(node_j.split('_')[1])
                    self.entanglement_matrix[i_idx, j_idx] = entanglement
                    self.entanglement_matrix[j_idx, i_idx] = entanglement
    
    def _register_quantum_nodes(self):
        """
        Register nodes with each other based on network topology.
        """
        for node_id, node in self.nodes.items():
            # Get neighbors from network graph
            neighbors = list(self.network_graph.neighbors(node_id))
            
            # Register each neighbor with the node
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                
                # Exchange public keys
                node.known_nodes[neighbor_id] = neighbor.public_key
                node.recognized_nodes.add(neighbor_id)
                
                # Set up entanglement pairs based on edge weight
                if self.network_graph.has_edge(node_id, neighbor_id):
                    entanglement = self.network_graph[node_id][neighbor_id]['weight']
                    node.entanglement_pairs[neighbor_id] = entanglement
    
    def submit_transaction(self, transaction_data):
        """
        Submit a transaction to the network.
        
        Parameters:
        -----------
        transaction_data : dict
            Transaction data
            
        Returns:
        --------
        str
            Transaction ID
        """
        # Create transaction ID using quantum-enhanced hashing
        tx_data_str = json.dumps(transaction_data, sort_keys=True)
        tx_id = hashlib.sha256(tx_data_str.encode()).hexdigest()
        
        # Add quantum timestamp for verification
        transaction_data['quantum_timestamp'] = time.time()
        transaction_data['quantum_enhanced'] = True
        transaction_data['quantum_enhancement_level'] = self.quantum_enhancement_level
        
        # Store transaction
        self.transaction_pool[tx_id] = transaction_data
        
        # Add to pending transactions
        if tx_id not in self.transaction_ledger and tx_id not in self.pending_transactions:
            self.pending_transactions.append(tx_id)
            
        return tx_id
    
    def run_quantum_consensus_round(self):
        """
        Run a single round of the quantum consensus protocol.
        
        Returns:
        --------
        dict
            Results of the consensus round
        """
        self.current_round += 1
        round_id = self.current_round
        
        print(f"Starting quantum consensus round {round_id}...")
        
        # Phase 1: Bid Requests
        # Each node creates a bid request for a consensus set to verify transactions
        bid_requests = {}
        for node_id, node in self.nodes.items():
            # Only nodes with enough connections can bid
            min_connections = max(3, int(self.num_nodes * 0.25))
            if len(node.recognized_nodes) >= min_connections:
                # Select a subset of pending transactions for this bid
                max_tx = min(5, len(self.pending_transactions))
                if max_tx > 0:
                    num_tx = np.random.randint(1, max_tx + 1)
                    tx_ids = np.random.choice(self.pending_transactions, size=num_tx, replace=False).tolist()
                    
                    # Create quantum-enhanced bid request
                    bid_request = node.create_quantum_bid_request(
                        transaction_ids=tx_ids,
                        consensus_set_size=min(5, self.num_nodes // 2)
                    )
                    
                    bid_requests[node_id] = bid_request
        
        print(f"  {len(bid_requests)} quantum bid requests created")
        
        # Phase 2: Distribute Bids
        # Send bid requests to all nodes
        for receiver_id, receiver in self.nodes.items():
            for sender_id, bid_request in bid_requests.items():
                # Nodes only receive bids from nodes they recognize
                if sender_id in receiver.recognized_nodes:
                    # Add quantum entanglement effects to timestamps
                    entanglement = receiver.entanglement_pairs.get(sender_id, 0.0)
                    timestamp_shift = np.random.normal(0, 0.05) * (1.0 - entanglement)
                    timestamp = time.time() + timestamp_shift
                    
                    receiver.receive_quantum_message(sender_id, bid_request, timestamp)
        
        # Phase 3: Collect Random Seeds
        # Each node contributes a quantum random seed
        quantum_seeds = {}
        for node_id, node in self.nodes.items():
            quantum_seeds[node_id] = node.generate_quantum_seed().hex()
        
        # Phase 4: Compute Global Quantum Seed
        global_quantum_seed = None
        seed_collector_node = self.nodes[f"node_0"]  # Use first node as seed collector
        global_quantum_seed = seed_collector_node.compute_quantum_global_seed(quantum_seeds)
        
        # Phase 5: Determine Consensus Sets
        # Use global quantum seed to deterministically form consensus sets
        all_node_ids = list(self.nodes.keys())
        consensus_requests = list(bid_requests.values())
        
        consensus_sets = seed_collector_node.determine_quantum_consensus_sets(
            global_quantum_seed, all_node_ids, consensus_requests
        )
        
        print(f"  {len(consensus_sets)} quantum consensus sets formed")
        
        # Phase 6: Voting on Transactions
        # Each node in a consensus set votes on the transactions
        transaction_votes = {}
        for cs in consensus_sets:
            for node_id in cs['nodes']:
                node = self.nodes[node_id]
                
                # Node votes on transactions
                votes = node.vote_on_quantum_transactions(cs)
                
                # Store votes
                if node_id not in transaction_votes:
                    transaction_votes[node_id] = {}
                transaction_votes[node_id].update(votes)
        
        # Phase 7: Compliance Checking
        # Nodes vote on the compliance of other nodes
        compliance_votes = {}
        for node_id, node in self.nodes.items():
            node_votes = node.vote_on_quantum_compliance(round_id)
            
            for target_id, vote in node_votes.items():
                if target_id not in compliance_votes:
                    compliance_votes[target_id] = {}
                compliance_votes[target_id][node_id] = vote
        
        # Compute final compliance results
        compliance_results = seed_collector_node.compute_quantum_compliance_result(compliance_votes)
        
        # Phase 8: Generate Proofs of Consensus
        proofs_of_consensus = []
        for cs in consensus_sets:
            proof = seed_collector_node.generate_quantum_proof_of_consensus(
                round_id, cs, transaction_votes, compliance_results
            )
            proofs_of_consensus.append(proof)
        
        # Phase 9: Update Transaction Ledger
        newly_verified = []
        for node_id, node in self.nodes.items():
            verified = node.update_quantum_transaction_ledger(proofs_of_consensus)
            newly_verified.extend(verified)
        
        # Remove duplicates in newly verified
        newly_verified = list(set(newly_verified))
        
        # Move verified transactions from pool to ledger
        for tx_id in newly_verified:
            if tx_id in self.transaction_pool:
                self.transaction_ledger[tx_id] = self.transaction_pool[tx_id]
                
                # Remove from pending transactions
                if tx_id in self.pending_transactions:
                    self.pending_transactions.remove(tx_id)
        
        print(f"  {len(newly_verified)} transactions verified by quantum consensus")
        
        # Phase 10: Update Quantum Entanglement
        # Interaction during consensus increases entanglement
        for cs in consensus_sets:
            cs_nodes = cs['nodes']
            for i, node_id1 in enumerate(cs_nodes):
                node1 = self.nodes[node_id1]
                
                for node_id2 in cs_nodes[i+1:]:
                    # Increase entanglement between nodes that participated together
                    entanglement_increase = 0.1 * self.quantum_enhancement_level
                    node1.update_entanglement_pairs(node_id2, entanglement_increase)
                    
                    # Update entanglement matrix
                    i_idx = int(node_id1.split('_')[1])
                    j_idx = int(node_id2.split('_')[1])
                    self.entanglement_matrix[i_idx, j_idx] += entanglement_increase
                    self.entanglement_matrix[j_idx, i_idx] += entanglement_increase
        
        # Store history
        self.consensus_sets_history[round_id] = consensus_sets
        self.compliance_history[round_id] = compliance_results
        self.transaction_votes_history[round_id] = transaction_votes
        
        # Return results
        results = {
            'round_id': round_id,
            'consensus_sets': consensus_sets,
            'compliance_results': compliance_results,
            'transaction_votes': transaction_votes,
            'proofs_of_consensus': proofs_of_consensus,
            'newly_verified': newly_verified,
            'quantum_enhancement_level': self.quantum_enhancement_level
        }
        
        return results
    
    def get_quantum_network_stats(self):
        """
        Get statistics about the quantum consensus network.
        
        Returns:
        --------
        dict
            Network statistics
        """
        # Calculate statistics
        verified_count = len(self.transaction_ledger)
        pending_count = len(self.pending_transactions)
        
        # Calculate average node honesty
        avg_honesty = sum(self.honesty_distribution) / len(self.honesty_distribution)
        
        # Calculate average quantum enhancement level
        avg_quantum_level = sum(node.quantum_enhancement_level for node in self.nodes.values()) / len(self.nodes)
        
        # Calculate average entanglement
        total_entanglement = 0
        entanglement_count = 0
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                total_entanglement += self.entanglement_matrix[i, j]
                entanglement_count += 1
                
        avg_entanglement = total_entanglement / entanglement_count if entanglement_count > 0 else 0
        
        # Return statistics
        return {
            'nodes': self.num_nodes,
            'verified_transactions': verified_count,
            'pending_transactions': pending_count,
            'total_transactions': verified_count + pending_count,
            'verification_rate': verified_count / (verified_count + pending_count) if (verified_count + pending_count) > 0 else 0,
            'consensus_rounds': self.current_round,
            'average_honesty': avg_honesty,
            'quantum_enhancement_level': self.quantum_enhancement_level,
            'average_quantum_enhancement': avg_quantum_level,
            'average_entanglement': avg_entanglement
        }


# Add hashlib import at the top which was missing
import hashlib
