#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consensus_network.py

Implementation of a Distributed Consensus Network (DCN) that manages multiple nodes,
facilitates communication between them, and oversees the consensus protocol.

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import time
import json
import networkx as nx
from consensus_node import ConsensusNode


class ConsensusNetwork:
    """
    A network of consensus nodes that collectively form a Distributed Consensus Network.
    Manages node communication, consensus set formation, and protocol execution.
    """
    
    def __init__(self, num_nodes=10, honesty_distribution=None, network_connectivity=0.8):
        """
        Initialize a consensus network.
        
        Parameters:
        -----------
        num_nodes : int, default=10
            Number of nodes in the network
        honesty_distribution : list or None, default=None
            List of honesty probabilities for each node (if None, all nodes are honest)
        network_connectivity : float, default=0.8
            Probability of connection between any two nodes
        """
        self.num_nodes = num_nodes
        
        # Set honesty distribution
        if honesty_distribution is None:
            self.honesty_distribution = [1.0] * num_nodes
        else:
            if len(honesty_distribution) != num_nodes:
                raise ValueError("honesty_distribution must have length equal to num_nodes")
            self.honesty_distribution = honesty_distribution
            
        # Create nodes
        self.nodes = {}
        for i in range(num_nodes):
            node_id = f"node_{i}"
            self.nodes[node_id] = ConsensusNode(node_id, self.honesty_distribution[i])
            
        # Create the network topology
        self.network_graph = self._create_network_topology(network_connectivity)
        
        # Initialize network attributes
        self.current_round = 0
        self.consensus_sets = []
        self.global_random_seed = None
        self.proofs_of_consensus = []
        self.transaction_ledger = []
        
        # Register nodes with each other
        self._register_nodes()
        
    def _create_network_topology(self, connectivity):
        """
        Create a network topology as a graph.
        
        Parameters:
        -----------
        connectivity : float
            Probability of connection between any two nodes
            
        Returns:
        --------
        networkx.Graph
            Network topology graph
        """
        # Create an Erdős-Rényi random graph
        node_ids = list(self.nodes.keys())
        G = nx.erdos_renyi_graph(len(node_ids), connectivity, seed=42)
        
        # Map node indices to node IDs
        mapping = {i: node_id for i, node_id in enumerate(node_ids)}
        G = nx.relabel_nodes(G, mapping)
        
        return G
    
    def _register_nodes(self):
        """
        Register nodes with each other based on network topology.
        """
        for node_id, node in self.nodes.items():
            # Get neighbors from topology
            neighbors = list(self.network_graph.neighbors(node_id))
            
            # Register each neighbor
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                node.known_nodes[neighbor_id] = neighbor.public_key_bytes
                node.recognized_nodes.add(neighbor_id)
    
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
        # Generate transaction ID from data
        tx_data_str = json.dumps(transaction_data, sort_keys=True)
        import hashlib
        tx_id = hashlib.sha256(tx_data_str.encode()).hexdigest()
        
        # Add transaction to the pool of each node
        for node in self.nodes.values():
            node.transaction_pool.append({
                'id': tx_id,
                'data': transaction_data,
                'timestamp': time.time()
            })
            
        return tx_id
    
    def run_consensus_round(self, transactions_per_bid=5, consensus_set_size=3):
        """
        Run a complete consensus round.
        
        Parameters:
        -----------
        transactions_per_bid : int, default=5
            Number of transactions per bid
        consensus_set_size : int, default=3
            Size of each consensus set
            
        Returns:
        --------
        dict
            Round results
        """
        self.current_round += 1
        round_id = self.current_round
        
        # 1. BID PHASE: Create bid requests from each node
        bid_requests = self._execute_bid_phase(transactions_per_bid, consensus_set_size)
        
        # 2. ACCEPT PHASE: Nodes accept or reject bids
        accepted_bids = self._execute_accept_phase(bid_requests)
        
        # 3. CONSENSUS PHASE: Determine consensus sets
        consensus_sets = self._execute_consensus_phase(accepted_bids)
        
        # 4. VOTE PHASE: Nodes vote on transactions
        transaction_votes = self._execute_vote_phase(consensus_sets)
        
        # 5. COMPLIANCE PHASE: Determine compliant nodes
        compliance_results = self._execute_compliance_phase(round_id)
        
        # 6. PROOF PHASE: Generate proofs of consensus
        proofs = self._generate_proofs_of_consensus(round_id, consensus_sets, 
                                                  transaction_votes, compliance_results)
        
        # 7. LEDGER PHASE: Update transaction ledger
        newly_verified = self._update_transaction_ledger(proofs)
        
        # Prepare round results
        results = {
            'round_id': round_id,
            'bid_requests': bid_requests,
            'accepted_bids': accepted_bids,
            'consensus_sets': consensus_sets,
            'transaction_votes': transaction_votes,
            'compliance_results': compliance_results,
            'proofs_of_consensus': proofs,
            'newly_verified': newly_verified
        }
        
        return results
    
    def _execute_bid_phase(self, transactions_per_bid, consensus_set_size):
        """
        Execute the bid phase of the consensus protocol.
        
        Parameters:
        -----------
        transactions_per_bid : int
            Number of transactions per bid
        consensus_set_size : int
            Size of each consensus set
            
        Returns:
        --------
        list
            Bid requests
        """
        bid_requests = []
        
        for node_id, node in self.nodes.items():
            # Skip if node has no transactions
            if not node.transaction_pool:
                continue
                
            # Select transactions for bid
            tx_ids = [tx['id'] for tx in node.transaction_pool[:transactions_per_bid]]
            
            # Create bid request
            bid_request = node.create_bid_request(tx_ids, consensus_set_size)
            bid_requests.append(bid_request)
            
            # Broadcast bid request to neighbors
            for neighbor_id in self.network_graph.neighbors(node_id):
                self.nodes[neighbor_id].receive_message(node_id, bid_request)
                
        return bid_requests
    
    def _execute_accept_phase(self, bid_requests):
        """
        Execute the accept phase of the consensus protocol.
        
        Parameters:
        -----------
        bid_requests : list
            Bid requests from nodes
            
        Returns:
        --------
        list
            Accepted bids
        """
        # For simplicity, all bids are accepted in this implementation
        # In a real implementation, nodes would verify signatures, check timestamps, etc.
        return bid_requests
    
    def _execute_consensus_phase(self, accepted_bids):
        """
        Execute the consensus phase of the consensus protocol.
        
        Parameters:
        -----------
        accepted_bids : list
            Accepted bids
            
        Returns:
        --------
        list
            Consensus sets
        """
        if not accepted_bids:
            return []
            
        # Collect random seeds from all nodes
        collected_seeds = {}
        for bid in accepted_bids:
            collected_seeds[bid['node_id']] = bid['random_seed']
            
        # Compute global random seed
        # Use the first node to compute it (all should get the same result)
        first_node = list(self.nodes.values())[0]
        self.global_random_seed = first_node.compute_global_random_seed(collected_seeds)
        
        # Determine consensus sets
        node_ids = list(self.nodes.keys())
        consensus_sets = first_node.determine_consensus_sets(
            self.global_random_seed, node_ids, accepted_bids
        )
        
        # Store consensus sets
        self.consensus_sets = consensus_sets
        
        # Inform nodes about their consensus sets
        for cs in consensus_sets:
            for node_id in cs['nodes']:
                # Add this consensus set to the node's sets
                self.nodes[node_id].consensus_sets.append(cs)
                
        return consensus_sets
    
    def _execute_vote_phase(self, consensus_sets):
        """
        Execute the vote phase of the consensus protocol.
        
        Parameters:
        -----------
        consensus_sets : list
            Consensus sets
            
        Returns:
        --------
        dict
            Transaction votes {node_id: {transaction_id: vote}}
        """
        transaction_votes = {}
        
        for cs in consensus_sets:
            for node_id in cs['nodes']:
                # Node votes on transactions
                votes = self.nodes[node_id].vote_on_transactions(cs)
                
                # Store votes
                transaction_votes[node_id] = votes
                
                # Inform other nodes in the consensus set about votes
                for other_node_id in cs['nodes']:
                    if other_node_id != node_id:
                        vote_message = {
                            'type': 'vote',
                            'round_id': self.current_round,
                            'node_id': node_id,
                            'consensus_set': cs['nodes'],
                            'votes': votes
                        }
                        self.nodes[other_node_id].receive_message(node_id, vote_message)
                        
        return transaction_votes
    
    def _execute_compliance_phase(self, round_id):
        """
        Execute the compliance phase of the consensus protocol.
        
        Parameters:
        -----------
        round_id : int
            Round ID
            
        Returns:
        --------
        dict
            Compliance results {node_id: is_compliant}
        """
        # Each node votes on compliance
        compliance_votes = {}
        for node_id, node in self.nodes.items():
            votes = node.vote_on_compliance(round_id)
            compliance_votes[node_id] = votes
            
            # Inform other nodes about compliance votes
            for neighbor_id in self.network_graph.neighbors(node_id):
                compliance_message = {
                    'type': 'compliance',
                    'round_id': round_id,
                    'node_id': node_id,
                    'votes': votes
                }
                self.nodes[neighbor_id].receive_message(node_id, compliance_message)
                
        # Compute compliance results
        # Use the first node to compute it (all should get the same result)
        first_node = list(self.nodes.values())[0]
        compliance_results = {}
        
        # Consolidate all votes
        all_votes = {}
        for voter_id, votes in compliance_votes.items():
            for node_id, vote in votes.items():
                if node_id not in all_votes:
                    all_votes[node_id] = {}
                all_votes[node_id][voter_id] = vote
                
        # Compute results
        compliance_results = first_node.compute_compliance_result(all_votes)
        
        return compliance_results
    
    def _generate_proofs_of_consensus(self, round_id, consensus_sets, 
                                      transaction_votes, compliance_results):
        """
        Generate proofs of consensus for verified transactions.
        
        Parameters:
        -----------
        round_id : int
            Round ID
        consensus_sets : list
            Consensus sets
        transaction_votes : dict
            {node_id: {transaction_id: vote}}
        compliance_results : dict
            {node_id: is_compliant}
            
        Returns:
        --------
        list
            Proofs of consensus
        """
        proofs = []
        
        for cs in consensus_sets:
            # Let each node in the consensus set generate a proof
            for node_id in cs['nodes']:
                if compliance_results.get(node_id, False):
                    node = self.nodes[node_id]
                    poc = node.generate_proof_of_consensus(
                        round_id, cs, transaction_votes, compliance_results
                    )
                    proofs.append(poc)
                    
        # Store proofs of consensus
        self.proofs_of_consensus.extend(proofs)
        
        return proofs
    
    def _update_transaction_ledger(self, proofs):
        """
        Update the transaction ledger based on proofs of consensus.
        
        Parameters:
        -----------
        proofs : list
            Proofs of consensus
            
        Returns:
        --------
        list
            Newly verified transactions
        """
        newly_verified = []
        
        # Update ledger of each node
        for node in self.nodes.values():
            verified = node.update_transaction_ledger(proofs)
            newly_verified.extend(verified)
            
        # Update global ledger
        for proof in proofs:
            for tx_id, verified in proof['transaction_results'].items():
                if verified and tx_id not in self.transaction_ledger:
                    self.transaction_ledger.append(tx_id)
                    
        return list(set(newly_verified))  # Remove duplicates
    
    def get_network_statistics(self):
        """
        Get statistics about the network.
        
        Returns:
        --------
        dict
            Network statistics
        """
        stats = {
            'num_nodes': self.num_nodes,
            'num_edges': self.network_graph.number_of_edges(),
            'average_degree': sum(dict(self.network_graph.degree()).values()) / self.num_nodes,
            'average_honesty': sum(self.honesty_distribution) / self.num_nodes,
            'num_transactions': len(set(tx for node in self.nodes.values() 
                                    for tx in node.transaction_pool)),
            'verified_transactions': len(self.transaction_ledger),
            'rounds_completed': self.current_round
        }
        
        return stats
