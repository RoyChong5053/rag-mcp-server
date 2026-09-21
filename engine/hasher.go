package engine

import (
	"hash/fnv"
)

// StringHash computes a hash for a string.
// This uses FNV-1a which is a common hash function.
// Note: Need to verify if this matches ST's calculateHash behavior.
func StringHash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}
