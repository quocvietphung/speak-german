"use client";
import React from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Heading,
  Wrap,
  WrapItem,
  Badge,
  Spinner,
  Card,
  Separator,
  ProgressCircle,
} from "@chakra-ui/react";

interface ScoreFeedbackProps {
  loading: boolean;
  score: number | null;
  mistakes: string[];
  tip: string;
  teacherFeedback: string;
}

export default class ScoreFeedbackCard extends React.Component<ScoreFeedbackProps> {
  render() {
    const { loading, score, mistakes, tip, teacherFeedback } = this.props;

    return (
      <Card.Root p={8} rounded="2xl" shadow="2xl" borderWidth="1px" bgGradient="linear(to-br, blue.50, white)">
        <Card.Header mb={4}>
          <Heading size="lg" color="blue.600">📊 Pronunciation Feedback</Heading>
        </Card.Header>

        <Card.Body>
          {loading ? (
            <VStack py={8} gap={4}>
              <Spinner size="xl" color="blue.500" />
              <Text fontSize="lg" color="gray.600">Analyzing your speech...</Text>
            </VStack>
          ) : score !== null ? (
            <VStack align="stretch" gap={6}>
              <HStack justify="center">
                <ProgressCircle.Root value={score} size="lg">
                  <ProgressCircle.Circle>
                    <ProgressCircle.Track />
                    <ProgressCircle.Range stroke={score >= 80 ? "green.400" : "orange.400"} />
                  </ProgressCircle.Circle>
                  <ProgressCircle.ValueText fontSize="2xl" fontWeight="bold">
                    {score}%
                  </ProgressCircle.ValueText>
                </ProgressCircle.Root>
              </HStack>

              <Box>
                <Text fontWeight="semibold" fontSize="lg" color="gray.700">❌ Mistake Words</Text>
                <Wrap mt={2} gap={2}>
                  {mistakes.length > 0 ? (
                    mistakes.map((w, i) => (
                      <WrapItem key={`${w}-${i}`}>
                        <Badge px={3} py={1} rounded="lg" fontSize="md" colorScheme="red">{w}</Badge>
                      </WrapItem>
                    ))
                  ) : (
                    <Text color="green.500" fontWeight="medium">Perfect! No mistakes 🎉</Text>
                  )}
                </Wrap>
              </Box>

              <Separator />

              <Box>
                <Text fontWeight="semibold" fontSize="lg" color="gray.700">💡 Tip</Text>
                <Text mt={1} color="gray.600">{tip}</Text>
              </Box>

              {teacherFeedback && (
                <>
                  <Separator />
                  <Box>
                    <Text fontWeight="semibold" fontSize="lg" color="gray.700">👩‍🏫 Teacher Feedback</Text>
                    <Text mt={1} whiteSpace="pre-wrap" color="gray.600">{teacherFeedback}</Text>
                  </Box>
                </>
              )}
            </VStack>
          ) : (
            <Text color="gray.400" py={8} textAlign="center" fontSize="lg">
              No score yet. Record and analyze your speech.
            </Text>
          )}
        </Card.Body>
      </Card.Root>
    );
  }
}