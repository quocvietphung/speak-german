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
      bgGradient="linear(to-b, gray.50, white)"
      p={{ base: 4, md: 8 }}
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
        <Card.Root
          p={{ base: 4, md: 6 }}
          rounded="2xl"
          shadow="sm"
          borderWidth="1px"
          bg="white"
          variant="elevated"
        >
          <Card.Header pb={2}>
            <Heading size="md">🎤 Pronunciation Practice</Heading>
          </Card.Header>
          <Card.Body>
            <VStack align="start" gap={5} w="full">
              <Text fontSize="sm" color="gray.600">
                Read the sentence aloud:
              </Text>

              <Text fontSize="xl" fontWeight="semibold" lineHeight="tall">
                {targetText}
              </Text>

              <HStack gap={4}>
                <Button
                  onClick={handleRecord}
                  colorScheme={recording ? "red" : "gray"}
                  rounded="full"
                  w="56px"
                  h="56px"
                  fontSize="xl"
                  shadow="xs"
                  _active={{ transform: "scale(0.98)" }}
                >
                  🎙
                </Button>
                <Text fontSize="sm" color="gray.500">
                  {recording ? "Recording…" : "00:03 s"}
                </Text>
              </HStack>

              <VStack align="start" gap={2} w="full">
                <Text fontSize="sm" fontWeight="medium" color="gray.700">
                  Auto-transcribed text
                </Text>
                <Textarea
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  rows={3}
                  resize="vertical"
                />
              </VStack>

              <Button
                colorScheme="blue"
                w="full"
                onClick={handleAnalyze}
                shadow="xs"
              >
                Analyze & Score
              </Button>
            </VStack>
          </Card.Body>
        </Card.Root>

        {/* Right: Score & Feedback */}
        <Card.Root
          p={{ base: 4, md: 6 }}
          rounded="2xl"
          shadow="sm"
          borderWidth="1px"
          bg="white"
          variant="elevated"
        >
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
                  <Text fontSize="sm" fontWeight="medium" color="gray.700">
                    Mistake words
                  </Text>
                  <Wrap gap={2}>
                    {mistakes.map((w, i) => (
                      <WrapItem key={i}>
                        <Badge colorScheme="red" px={2} py={1} rounded="md">
                          {w}
                        </Badge>
                      </WrapItem>
                    ))}
                  </Wrap>
                </VStack>

                <Separator />

                <VStack align="start" gap={2} w="full">
                  <Text fontSize="sm" fontWeight="medium" color="gray.700">
                    Tip
                  </Text>
                  <Text fontSize="sm" color="gray.800">
                    {tip}
                  </Text>
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
