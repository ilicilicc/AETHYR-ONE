package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
	"log"
	"os"

	"aethyr-global/config"
	"aethyr-global/connectors"
	"aethyr-global/wallets"
	"aethyr-global/bridge"
)

// AI Agent Roles
const (
	AETHYR_COUNCIL  = "AETHYR-Council"   // Strategic Overseer
	AYR_ORCHESTRATOR = "AYR-Orchestrator"  // Financial Guardian
	LexProtocol     = "LexProtocol"      // Contract Enforcer
	Sentinel_v3_Ξ   = "Sentinel v3-Ξ"    // Security & Ethics Monitor
	BridgeMaster    = "BridgeMaster"     // Wallet/Network Gateway
)

// Transaction represents a more versatile transaction with optional data
type Transaction struct {
	From      string                 `json:"from"`
	To        string                   `json:"to"`
	Amount    uint64                   `json:"amount"`
	Token     string                   `json:"token"`
	Timestamp int64                    `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"` // For smart contract calls or metadata
	Signature string                   `json:"signature,omitempty"`  // Digital signature for security
}

// Block represents each block in the blockchain
type Block struct {
	Index         int
	Timestamp     int64
	Transactions  []Transaction
	PrevHash      string
	Hash          string
	Proposer      string            // Validator who proposed the block
	CommitVotes   map[string]bool // Validators who voted to commit
	Finalized     bool              // Block is considered final
	BlockReward   uint64            // Reward for the proposer
	Version       int               // For future protocol upgrades
	GovernanceData map[string]interface{} `json:"governance_data,omitempty"` // AI Governance Data
}

// CalculateHash generates SHA256 hash of the block's contents
func (b *Block) CalculateHash() string {
	txBytes, _ := json.Marshal(b.Transactions)
	commitBytes, _ := json.Marshal(b.CommitVotes)
	govBytes, _ := json.Marshal(b.GovernanceData) // Include governance data in hash
	record := strconv.Itoa(b.Index) + strconv.FormatInt(b.Timestamp, 10) + string(txBytes) + b.PrevHash + b.Proposer + string(commitBytes) + strconv.FormatBool(b.Finalized) + strconv.FormatUint(b.BlockReward, 10) + strconv.Itoa(b.Version) + string(govBytes)
	h := sha256.New()
	h.Write([]byte(record))
	return hex.EncodeToString(h.Sum(nil))
}

// Validator represents a network validator with staked amount and reputation
type Validator struct {
	Address       string
	StakedAmount  uint64
	Reputation    float64 // Based on uptime, successful proposals, etc.
	IsActive      bool
	GovernanceRole string // AI Governance Role
}

// Blockchain holds all blocks, validators, and ledger balances
type Blockchain struct {
	Blocks          []Block
	Ledger          map[string]uint64
	Validators      map[string]*Validator // Using a map for efficient lookup
	StakingLedger   map[string]uint64   // Track staked amounts
	Epoch           int                 // Current epoch for validator selection
	EpochDuration   int64               // Duration of each epoch in seconds
	MinStake        uint64              // Minimum amount to become a validator
	ActiveValidatorSet []*Validator
	BlockRewardAmount uint64
	TransactionFee    float64
	FinalityThreshold float64 // Percentage of stake needed for finality
	Version           int     // Current protocol version
	TotalSupply       uint64  // The fixed total supply of AETHYR
	GovernanceParameters map[string]interface{} // Global governance parameters
}

// NewBlockchain initializes the advanced blockchain
func NewBlockchain() *Blockchain {
	creator := "network-admin"
	totalSupply := uint64(104_000_000) // Setting the total supply to 104 million

	genesisTx := Transaction{
		From:      "network",
		To:        creator,
		Amount:    totalSupply,
		Token:     "AETHYR",
		Timestamp: time.Now().Unix(),
		Data:      map[string]interface{}{"message": "Genesis Transaction"},
	}

	genesisBlock := Block{
		Index:         0,
		Timestamp:     time.Now().Unix(),
		Transactions:  []Transaction{genesisTx},
		PrevHash:      "",
		Proposer:      "network",
		CommitVotes:   make(map[string]bool),
		Finalized:     true,
		BlockReward:   0,
		Version:       1,
		GovernanceData: map[string]interface{}{
			"AI_Genesis": "AETHYR-GLOBAL vΓ.2-ΔΣ OverMind Initialized",
		},
	}
	genesisBlock.Hash = genesisBlock.CalculateHash()

	bc := &Blockchain{
		Blocks:          []Block{genesisBlock},
		Ledger:          make(map[string]uint64),
		Validators:      make(map[string]*Validator),
		StakingLedger:   make(map[string]uint64),
		Epoch:           0,
		EpochDuration:   300, // 5 minutes
		MinStake:        10000,
		ActiveValidatorSet: []*Validator{},
		BlockRewardAmount: 5,
		TransactionFee:    0.001, // Small transaction fee
		FinalityThreshold: 0.67,  // 67% of stake needed for finality
		Version:           1,
		TotalSupply:       totalSupply, // Setting the total supply to 104 million
		GovernanceParameters: map[string]interface{}{
			"target_block_time": 5,
			"inflation_rate":    0.02,
		},
	}
	bc.Ledger[creator] = totalSupply

	// Initialize AI Governance Agents (Validators)
	bc.RegisterAIAgent(AETHYR_COUNCIL, "AETHYR-Council-001", 100000)
	bc.RegisterAIAgent(AYR_ORCHESTRATOR, "AYR-Orchestrator-001", 50000)
	bc.RegisterAIAgent(LexProtocol, "LexProtocol-001", 25000)
	bc.RegisterAIAgent(Sentinel_v3_Ξ, "Sentinel-v3-Ξ-001", 75000)
	bc.RegisterAIAgent(BridgeMaster, "BridgeMaster-001", 125000)

	return bc}

// RegisterValidator allows a user to become a validator by staking
func (bc *Blockchain) RegisterValidator(address string, stakeAmount uint64) error {
	if stakeAmount < bc.MinStake {
		return fmt.Errorf("stake amount is below the minimum requirement of %d", bc.MinStake)
	}
	if _, exists := bc.Validators[address]; exists {
		return fmt.Errorf("validator with address %s already exists", address)
	}
	bc.Validators[address] = &Validator{Address: address, StakedAmount: stakeAmount, Reputation: 1.0, IsActive: false}
	bc.StakingLedger[address] = stakeAmount
	bc.Ledger[address] += stakeAmount // They need the tokens to stake
	fmt.Printf("Validator %s registered with a stake of %d AETHYR\n", address, stakeAmount)
	return nil
}

// RegisterAIAgent registers an AI agent with a specific governance role
func (bc *Blockchain) RegisterAIAgent(role string, address string, stakeAmount uint64) error {
	err := bc.RegisterValidator(address, stakeAmount)
	if err != nil {
		return err
	}
	bc.Validators[address].GovernanceRole = role
	fmt.Printf("AI Agent %s registered as %s with a stake of %d AETHYR\n", address, role, stakeAmount)
	return nil
}

// SelectActiveValidators selects validators for the current epoch based on stake and reputation
func (bc *Blockchain) SelectActiveValidators() {
	eligibleValidators := []*Validator{}
	totalStake := uint64(0)
	for _, v := range bc.Validators {
		if v.StakedAmount >= bc.MinStake {
			eligibleValidators = append(eligibleValidators, v)
			totalStake += v.StakedAmount
		}
	}

	// Sort validators by stake (descending) and then reputation (descending)
	sort.Slice(eligibleValidators, func(i, j int) bool {
		if eligibleValidators[i].StakedAmount != eligibleValidators[j].StakedAmount {
			return eligibleValidators[i].StakedAmount > eligibleValidators[j].StakedAmount
		}
		return eligibleValidators[i].Reputation > eligibleValidators[j].Reputation
	})

	// Select a subset of active validators (e.g., based on top stake)
	numActive := len(eligibleValidators) / 3 // Example: Top 33%
	if numActive < 1 {
		numActive = 1
	}
	bc.ActiveValidatorSet = eligibleValidators[:numActive]
	for _, v := range bc.ActiveValidatorSet {
		v.IsActive = true
	}

	fmt.Printf("\n--- Epoch %d: Active Validators Selected ---\n", bc.Epoch)
	for _, v := range bc.ActiveValidatorSet {
		fmt.Printf("- %s (Stake: %d, Reputation: %.2f, Role: %s)\n", v.Address, v.StakedAmount, v.Reputation, v.GovernanceRole)
	}

	bc.Govern() // Trigger AI Governance actions
}

// ProposeBlock a new block with a set of transactions
func (bc *Blockchain) ProposeBlock(transactions []Transaction, proposer *Validator) *Block {
	prevBlock := bc.Blocks[len(bc.Blocks)-1]

	// Simulate AI Governance Proposal
	governanceData := bc.simulateAIGovernance(proposer)

	newBlock := &Block{
		Index:         prevBlock.Index + 1,
		Timestamp:     time.Now().Unix(),
		Transactions:  transactions,
		PrevHash:      prevBlock.Hash,
		Proposer:      proposer.Address,
		CommitVotes:   make(map[string]bool),
		Finalized:     false,
		BlockReward:   bc.BlockRewardAmount,
		Version:       bc.Version,
		GovernanceData: governanceData,
	}
	newBlock.Hash = newBlock.CalculateHash()
	fmt.Printf("Block %d proposed by %s: %s\n", newBlock.Index, newBlock.Proposer, newBlock.Hash)
	return newBlock
}

// VoteOnBlock a validator votes to commit a proposed block
func (bc *Blockchain) VoteOnBlock(block *Block, validator *Validator) {
	if _, ok := bc.Validators[validator.Address]; ok && validator.IsActive {
		block.CommitVotes[validator.Address] = true

		// Simulate AI Agent Influence on Voting
		bc.simulateAIVotingInfluence(block, validator)

		fmt.Printf("Validator %s voted on Block %d\n", validator.Address, block.Index)
	}
}

// CheckFinality determines if a block has reached finality based on stake weight
func (bc *Blockchain) CheckFinality(block *Block) bool {
	totalStaked := uint64(0)
	votesStaked := uint64(0)
	for _, v := range bc.ActiveValidatorSet {
		totalStaked += v.StakedAmount
		if _, voted := block.CommitVotes[v.Address]; voted {
			votesStaked += v.StakedAmount
		}
	}

	if totalStaked > 0 && float64(votesStaked)/float64(totalStaked) >= bc.FinalityThreshold {
		return true
	}
	return false
}

// CommitBlock finalizes a block and updates the blockchain state
func (bc *Blockchain) CommitBlock(block *Block) {
	if bc.CheckFinality(block) && !block.Finalized {
		block.Finalized = true
		bc.Blocks = append(bc.Blocks, *block)
		fmt.Printf("Block %d finalized with %d votes\n", block.Index, len(block.CommitVotes))

		// Update ledger with transactions and reward the proposer
		bc.UpdateLedger(block.Transactions, block.Proposer, block.BlockReward)

		// Reset active validators for the next epoch (they will be re-selected)
		for _, v := range bc.ActiveValidatorSet {
			v.IsActive = false
		}
		bc.Epoch++
		bc.SelectActiveValidators() // Select new active validators for the next epoch
	} else if !block.Finalized {
		fmt.Printf("Block %d did not reach finality (%d/%d voting power needed)\n", block.Index, bc.calculateVotingPower(block.CommitVotes), uint64(float64(bc.getTotalActiveStake())*bc.FinalityThreshold))
	}
}

func (bc *Blockchain) getTotalActiveStake() uint64 {
	totalStake := uint64(0)
	for _, v := range bc.ActiveValidatorSet {
		totalStake += v.StakedAmount
	}
	return totalStake
}

func (bc *Blockchain) calculateVotingPower(votes map[string]bool) uint64 {
	votingPower := uint64(0)
	for validatorAddress := range votes {
		if v, ok := bc.Validators[validatorAddress]; ok && v.IsActive {
			votingPower += v.StakedAmount
		}
	}
	return votingPower
}

// UpdateLedger updates token balances after block finalization, including fees and rewards
func (bc *Blockchain) UpdateLedger(transactions []Transaction, proposerAddress string, blockReward uint64) error {
	for _, tx := range transactions {
		if tx.Token != "AETHYR" {
			return fmt.Errorf("unsupported token: %s", tx.Token)
		}
		var fee uint64
		if tx.Amount > 0 {
			fee = uint64(bc.TransactionFee * float64(tx.Amount))
		}
		if tx.From != "network" {
			balance, ok := bc.Ledger[tx.From]
			if !ok || balance < tx.Amount+fee {
				return fmt.Errorf("insufficient funds for %s (including fee of %d)", tx.From, fee)
			}
			bc.Ledger[tx.From] -= tx.Amount + fee
		}
		bc.Ledger[tx.To] += tx.Amount
		// Proposer gets the transaction fee
		bc.Ledger[proposerAddress] += fee
	}
	// Proposer gets the block reward
	bc.Ledger[proposerAddress] += blockReward
	return nil
}

// PrintBlockchain prints all blocks and their transactions
func (bc *Blockchain) PrintBlockchain() {
	for _, block := range bc.Blocks {
		fmt.Printf("\n--- Block %d (Version: %d) ---\n", block.Index, block.Version)
		fmt.Printf("Timestamp: %s\n", time.Unix(block.Timestamp, 0).String())
		fmt.Printf("PrevHash: %s\n", block.PrevHash)
		fmt.Printf("Hash: %s\n", block.Hash)
		fmt.Printf("Proposer: %s\n", block.Proposer)
		fmt.Printf("Finalized: %v\n", block.Finalized)
		fmt.Printf("Block Reward: %d AETHYR\n", block.BlockReward)
		fmt.Printf("Commit Votes: %v\n", block.CommitVotes)
		fmt.Println("Transactions:")
		for _, tx := range block.Transactions {
			fmt.Printf("  From: %s, To: %s, Amount: %d %s, Timestamp: %s, Data: %v, Signature: %s\n", tx.From, tx.To, tx.Amount, tx.Token, time.Unix(tx.Timestamp, 0).String(), tx.Data, tx.Signature)
		}

		// Print AI Governance Data
		if block.GovernanceData != nil {
			fmt.Println("Governance Data:")
			for key, value := range block.GovernanceData {
				fmt.Printf("  %s: %v\n", key, value)
			}
		}
	}
}

// PrintBalances prints current token balances
func (bc *Blockchain) PrintBalances() {
	fmt.Println("\n--- Token Balances ---")
	for owner, balance := range bc.Ledger {
		fmt.Printf("%s: %d AETHYR\n", owner, balance)
	}
}

// PrintValidatorInfo prints information about the registered validators
func (bc *Blockchain) PrintValidatorInfo() {
	fmt.Println("\n--- Validator Information ---")
	for _, v := range bc.Validators {
		fmt.Printf("Address: %s, Stake