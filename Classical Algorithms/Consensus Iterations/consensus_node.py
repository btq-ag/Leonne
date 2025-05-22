#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consensus_node.py

Implementation of a node in a Distributed Consensus Network (DCN).
Each node can participate in consensus sets, submit and verify transactions,
and engage in the DCN protocol phases: bid, accept, consensus, and compliance.

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


class ConsensusNode:
    """
    A node in a Distributed Consensus Network that participates in 
    the consensus protocol and transaction verification.
    """
    
    def __init__(self, node_id, honesty_probability=1.0):
        """
        Initialize a consensus node.
        
        Parameters:
        -----------
        node_id : int or str
            Unique identifier for the node
        honesty_probability : float, default=1.0
            Probability that the node behaves honestly (for simulation purposes)
        """
        self.node_id = node_id
        self.honesty_probability = honesty_probability
        self.is_compliant = True
        
        # Generate RSA key pair for the node
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        
        # Serialized public key for sharing
        self.public_key_bytes = self.public_key.public_bytes(
            encoding=Encoding.PEM, 
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        # Node's view of the network
        self.known_nodes = {}  # {node_id: public_key}
        self.recognized_nodes = set()  # Set of node_ids recognized by this node
        
        # Consensus-related attributes
        self.consensus_sets = []  # List of consensus sets this node is part of
        self.bid_requests = []  # List of bid requests from this node
        self.transaction_verifications = {}  # {tx_id: verification_result}
        
        # Protocol state
        self.current_round = 0
        self.messages_received = {}  # {round_id: {node_id: message}}
        self.message_timestamps = {}  # {round_id: {node_id: timestamp}}
        
        # Compliance tracking
        self.compliance_votes = {}  # {round_id: {node_id: {voter_id: vote}}}
        
        # Transaction pool and ledger
        self.transaction_pool = []
        self.verified_transactions = []
        
        # Random seeds
        self.local_random_seed = None
    
    def generate_random_seed(self, transaction_id=None):
        """
        Generate a local random seed based on transaction ID or current time.
        
        Parameters:
        -----------
        transaction_id : str, optional
            Transaction ID to use for random seed generation
            
        Returns:
        --------
        bytes
            Random seed as bytes
        """
        if transaction_id is None:
            # Use current time if no transaction ID provided
            data = f"{self.node_id}_{time.time()}"
        else:
            data = f"{self.node_id}_{transaction_id}"
            
        self.local_random_seed = hashlib.sha256(data.encode()).digest()
        return self.local_random_seed
    
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
    
    def create_bid_request(self, transaction_ids, consensus_set_size):
        """
        Create a bid request for consensus on transactions.
        
        Parameters:
        -----------
        transaction_ids : list
            List of transaction IDs to request consensus for
        consensus_set_size : int
            Requested size of the consensus set
            
        Returns:
        --------
        dict
            Bid request
        """
        # Act dishonestly with probability (1 - honesty_probability)
        is_honest = np.random.random() < self.honesty_probability
        
        bid_request = {
            'node_id': self.node_id,
            'round_id': self.current_round,
            'transaction_ids': transaction_ids,
            'consensus_set_size': consensus_set_size,
            'recognized_nodes': list(self.recognized_nodes),
            'timestamp': time.time(),
            'random_seed': self.generate_random_seed().hex()
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
    
    def receive_message(self, sender_id, message, timestamp=None):
        """
        Receive and process a message from another node.
        
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
        
        # Store message and timestamp
        self.messages_received[round_id][sender_id] = message
        self.message_timestamps[round_id][sender_id] = timestamp
        
        return True
    
    def vote_on_compliance(self, round_id, tolerance=1.0):
        """
        Vote on the compliance of nodes based on message timing.
        
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
        if round_id not in self.message_timestamps:
            return {}
            
        timestamps = self.message_timestamps[round_id]
        if not timestamps:
            return {}
            
        # Calculate median timestamp
        median_time = np.median(list(timestamps.values()))
        
        # Vote on compliance based on timing
        compliance_votes = {}
        for node_id, timestamp in timestamps.items():
            # Node is compliant if its timestamp is within tolerance of median
            is_compliant = abs(timestamp - median_time) <= tolerance
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
    
    def verify_transaction(self, transaction_id, honest_vote=True):
        """
        Verify a transaction (simulated).
        
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
        # In a real implementation, this would involve checking the transaction
        # against the ledger, validating signatures, etc.
        
        # For simulation purposes, vote honestly with probability honesty_probability
        is_honest = np.random.random() < self.honesty_probability
        
        if is_honest:
            result = honest_vote
        else:
            # If dishonest, vote randomly
            result = np.random.choice([True, False])
            
        self.transaction_verifications[transaction_id] = result
        return result
    
    def compute_global_random_seed(self, collected_seeds):
        """
        Compute global random seed from collected local seeds.
        
        Parameters:
        -----------
        collected_seeds : dict
            {node_id: seed} from all participating nodes
            
        Returns:
        --------
        bytes
            Global random seed
        """
        # Sort seeds by node_id to ensure deterministic result
        sorted_seeds = [collected_seeds[node_id] for node_id in sorted(collected_seeds.keys())]
        
        # Concatenate and hash
        concatenated = b''.join([bytes.fromhex(seed) if isinstance(seed, str) else seed for seed in sorted_seeds])
        global_seed = hashlib.sha256(concatenated).digest()
        
        return global_seed
    
    def determine_consensus_sets(self, global_seed, node_ids, consensus_requests):
        """
        Determine the consensus sets based on the global random seed.
        
        Parameters:
        -----------
        global_seed : bytes
            Global random seed
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
          # Generate deterministic random generator
        # Convert bytes to int and make sure it's within the valid range for RandomState
        seed_int = int.from_bytes(global_seed, byteorder='big')
        seed_int = seed_int % (2**32 - 1)  # Ensure seed is within valid range (0 to 2^32 - 1)
        rng = np.random.RandomState(seed_int)
        
        consensus_sets = []
        remaining_nodes = node_ids.copy()
        
        for request in consensus_requests:
            # Get requested consensus set size
            set_size = min(request['consensus_set_size'], len(remaining_nodes))
            
            if set_size == 0:
                continue
                
            # Randomly select nodes for this consensus set
            selected_indices = rng.choice(len(remaining_nodes), size=set_size, replace=False)
            consensus_set = [remaining_nodes[i] for i in selected_indices]
            
            # Remove selected nodes from remaining nodes
            remaining_nodes = [node for i, node in enumerate(remaining_nodes) if i not in selected_indices]
            
            consensus_sets.append({
                'request_node_id': request['node_id'],
                'transaction_ids': request['transaction_ids'],
                'nodes': consensus_set
            })
            
            # If no more nodes, break
            if not remaining_nodes:
                break
                
        return consensus_sets
    
    def vote_on_transactions(self, consensus_set):
        """
        Vote on transactions in a consensus set.
        
        Parameters:
        -----------
        consensus_set : dict
            Consensus set with transaction IDs
            
        Returns:
        --------
        dict
            Votes on transactions {transaction_id: bool}
        """
        votes = {}
        for tx_id in consensus_set['transaction_ids']:
            votes[tx_id] = self.verify_transaction(tx_id)
            
        return votes
    
    def compute_compliance_result(self, compliance_votes):
        """
        Compute the final compliance result based on votes.
        
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
            # Count votes
            vote_count = sum(1 for v in votes.values() if v)
            # Node is compliant if majority votes yes
            result[node_id] = vote_count > len(votes) / 2
            
        return result
    
    def generate_proof_of_consensus(self, round_id, consensus_set, transaction_votes, compliance_result):
        """
        Generate proof of consensus for verified transactions.
        
        Parameters:
        -----------
        round_id : int
            Round ID
        consensus_set : dict
            Consensus set details
        transaction_votes : dict
            {node_id: {transaction_id: vote}}
        compliance_result : dict
            {node_id: is_compliant}
            
        Returns:
        --------
        dict
            Proof of consensus
        """
        # Filter out non-compliant nodes
        compliant_nodes = [node_id for node_id in consensus_set['nodes'] 
                          if compliance_result.get(node_id, False)]
          # Calculate transaction results
        transaction_results = {}
        for tx_id in consensus_set['transaction_ids']:
            # Count votes for this transaction from compliant nodes
            votes = [transaction_votes.get(node_id, {}).get(tx_id, False) 
                    for node_id in compliant_nodes]
            
            # Transaction is verified if majority votes yes
            if votes:
                # Convert to Python bool to avoid NumPy bool_ which isn't JSON serializable
                transaction_results[tx_id] = bool(sum(votes) > len(votes) / 2)
            else:
                transaction_results[tx_id] = False
        
        # Create proof of consensus
        poc = {
            'round_id': round_id,
            'timestamp': time.time(),
            'request_node_id': consensus_set['request_node_id'],
            'consensus_set': consensus_set['nodes'],
            'compliant_nodes': compliant_nodes,
            'transaction_results': transaction_results,
            'signature': {}  # Will be filled with signatures
        }
          # Sign the proof
        # Convert all values to Python native types to ensure JSON serialization works
        json_safe_poc = {k: (v if k != 'transaction_results' else {tx: bool(result) for tx, result in v.items()}) 
                         for k, v in poc.items() if k != 'signature'}
        message = json.dumps(json_safe_poc, sort_keys=True).encode()
        poc['signature'][self.node_id] = self.sign_message(message).hex()
        
        return poc
    
    def update_transaction_ledger(self, proofs_of_consensus):
        """
        Update the transaction ledger based on proofs of consensus.
        
        Parameters:
        -----------
        proofs_of_consensus : list
            List of proofs of consensus
            
        Returns:
        --------
        list
            Newly verified transactions
        """
        newly_verified = []
        
        for poc in proofs_of_consensus:
            for tx_id, verified in poc['transaction_results'].items():
                if verified and tx_id not in self.verified_transactions:
                    self.verified_transactions.append(tx_id)
                    newly_verified.append(tx_id)
                    
        return newly_verified
