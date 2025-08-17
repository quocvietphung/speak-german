"use client";

import React from "react";
import {
  Box, VStack, HStack, Text, Heading, Wrap, WrapItem, Badge,
  Spinner, Card, Separator, Button, Icon
} from "@chakra-ui/react";
import { MdPlayCircle } from "react-icons/md";

interface ScoreFeedbackProps {
  loading: boolean;
  score: number | null;
  mistakes: string[];
  tip: string;
  teacherFeedback: string;
  transcript: string;
}

export default function ScoreFeedbackCard({
  loading,
  score,
  mistakes,
  tip,
  teacherFeedback,
  transcript,
}: ScoreFeedbackProps) {
  const handleSpeak = () => {
    if ("speechSynthesis" in window && transcript) {
      const u = new SpeechSynthesisUtterance(transcript);
      u.lang = /[äöüÄÖÜß]/.test(transcript) ? "de-DE" : "en-US";
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
  };

  return (
    <Card.Root p={6} rounded="2xl" shadow="xl" borderWidth="1px"
      bgGradient="linear(to-br, white, gray.50)">
      <Card.Header>
        <Heading size="md">📊 Score & Feedback</Heading>
      </Card.Header>

      <Card.Body>
        {loading ? (
          <HStack justify="center" gap={4}>
            <Spinner size="lg" color="blue.400" />
            <Text>Analyzing your speech...</Text>
          </HStack>
        ) : score !== null ? (
          <VStack align="start" gap={5} w="full">
            <Text fontSize="4xl" fontWeight="bold"
              color={score >= 80 ? "green.500" : "orange.400"}>
              {score}%
            </Text>

            {/* **AI-parsed sentence playback** */}
            <Box w="full">
              <Text fontWeight="medium" mb={2}>
                Your sentence (AI detected)
              </Text>
              {transcript ? (
                <HStack gap={3}>
                  <Button colorScheme="teal" onClick={handleSpeak}>
                    <Icon as={MdPlayCircle} />
                    <Text ml={2}>Play transcript</Text>
                  </Button>
                  <Text fontStyle="italic" color="gray.600">
                    "{transcript}"
                  </Text>
                </HStack>
              ) : (
                <Text color="gray.400">No transcript yet. Record first.</Text>
              )}
            </Box>

            <Separator orientation="horizontal" />

            <Box w="full">
              <Text fontWeight="medium">Mistake Words</Text>
              <Wrap mt={2} gap={2}>
                {mistakes.length > 0 ? (
                  mistakes.map((w, i) => (
                    <WrapItem key={`${w}-${i}`}>
                      <Badge colorPalette="red">{w}</Badge>
                    </WrapItem>
                  ))
                ) : (
                  <Text color="gray.400">No mistakes 🎉</Text>
                )}
              </Wrap>
            </Box>

            <Separator orientation="horizontal" />

            <Box w="full">
              <Text fontWeight="medium">Tip</Text>
              <Text>{tip}</Text>
            </Box>

            {teacherFeedback && (
              <>
                <Separator orientation="horizontal" />
                <Box w="full">
                  <Text fontWeight="medium">👩‍🏫 Teacher Feedback</Text>
                  <Text whiteSpace="pre-wrap">{teacherFeedback}</Text>
                </Box>
              </>
            )}
          </VStack>
        ) : (
          <Text color="gray.400">No score yet. Record and analyze.</Text>
        )}
      </Card.Body>
    </Card.Root>
  );
}