"use client";

import { useState } from "react";
import {
  Box,
  Grid,
  VStack,
  HStack,
  Text,
  Button,
  Textarea,
  Badge,
  Heading,
  Separator,
  Wrap,
  WrapItem,
  Card,
} from "@chakra-ui/react";

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [mistakes, setMistakes] = useState<string[]>([]);
  const [tip, setTip] = useState("");
  const [transcript, setTranscript] = useState("");
  const targetText = "Ich möchte bitte eine Flasche Mineralwasser.";

  const handleRecord = () => setRecording((prev) => !prev);

  const handleAnalyze = async () => {
    // TODO: call your Flask API here and set state from response
    setScore(87);
    setMistakes(["michte", "vite"]);
    setTranscript("Ich michte vite eine Flasche mineralvasser");
    setTip('In „Ich“, the „ch“ sound should be pronounced like [x].');
  };

  return (
    <Box
      minH="100vh"
      bg="gray.50"
      p={{ base: 4, md: 6 }}
      display="flex"
      alignItems="center"
      justifyContent="center"
    >
      <Grid
        templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }}
        gap={{ base: 6, md: 8 }}
        maxW="6xl"
        w="100%"
      >
        {/* Left: Pronunciation Practice */}
        <Card.Root p={4} variant="elevated">
          <Card.Header pb={2}>
            <Heading size="md">🎤 Pronunciation Practice</Heading>
          </Card.Header>
          <Card.Body>
            <VStack align="start" gap={5} w="full">
              <Text fontSize="sm" color="gray.500">
                Read the sentence aloud:
              </Text>

              <Text fontSize="xl" fontWeight="bold">
                {targetText}
              </Text>

              <HStack gap={4}>
                <Button
                  onClick={handleRecord}
                  colorScheme={recording ? "red" : "gray"}
                  borderRadius="full"
                  w="48px"
                  h="48px"
                >
                  🎙
                </Button>
                <Text fontSize="sm" color="gray.500">
                  00:03 s
                </Text>
              </HStack>

              <VStack align="start" gap={2} w="full">
                <Text fontSize="sm" fontWeight="medium">
                  Auto-transcribed text
                </Text>
                <Textarea
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  rows={3}
                />
              </VStack>

              <Button colorScheme="blue" w="full" onClick={handleAnalyze}>
                Analyze & Score
              </Button>
            </VStack>
          </Card.Body>
        </Card.Root>

        {/* Right: Score & Feedback */}
        <Card.Root p={4} variant="elevated">
          <Card.Header pb={2}>
            <Heading size="md">📊 Score & Feedback</Heading>
          </Card.Header>
          <Card.Body>
            {score !== null ? (
              <VStack align="start" gap={5} w="full">
                <Box>
                  <Text fontSize="4xl" fontWeight="bold">
                    {score}%
                  </Text>
                  <Text fontSize="sm" color="gray.500">
                    WER 0.20 &nbsp; CER 0.06
                  </Text>
                </Box>

                <VStack align="start" gap={2} w="full">
                  <Text fontSize="sm" fontWeight="medium">
                    Mistake words
                  </Text>
                  <Wrap>
                    {mistakes.map((w, i) => (
                      <WrapItem key={i}>
                        <Badge colorScheme="red" px={2} py={1}>
                          {w}
                        </Badge>
                      </WrapItem>
                    ))}
                  </Wrap>
                </VStack>

                <Separator />

                <VStack align="start" gap={2} w="full">
                  <Text fontSize="sm" fontWeight="medium">
                    Tip
                  </Text>
                  <Text fontSize="sm">{tip}</Text>
                </VStack>
              </VStack>
            ) : (
              <Text fontSize="sm" color="gray.400">
                No score yet. Record and analyze to see feedback.
              </Text>
            )}
          </Card.Body>
        </Card.Root>
      </Grid>
    </Box>
  );
}
