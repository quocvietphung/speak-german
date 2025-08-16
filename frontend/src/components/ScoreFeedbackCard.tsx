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
  Separator,
  Card,
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
      <Card.Root p={6} rounded="2xl" shadow="sm" borderWidth="1px" bg="white">
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
            <VStack align="start" gap={5}>
              <Text
                fontSize="4xl"
                fontWeight="bold"
                color={score >= 80 ? "green.500" : "orange.400"}
              >
                {score}%
              </Text>

              {/* Mistakes */}
              <Box w="full">
                <Text fontWeight="medium">Mistake words</Text>
                <Wrap mt={2} gap={2}>
                  {mistakes.length > 0 ? (
                    mistakes.map((w, i) => (
                      <WrapItem key={`${w}-${i}`}>
                        <Badge colorScheme="red">{w}</Badge>
                      </WrapItem>
                    ))
                  ) : (
                    <Text color="gray.400">No mistakes 🎉</Text>
                  )}
                </Wrap>
              </Box>

              <Separator />

              {/* Tip */}
              <Box w="full">
                <Text fontWeight="medium">Tip</Text>
                <Text>{tip}</Text>
              </Box>

              {/* Teacher Feedback */}
              {teacherFeedback && (
                <>
                  <Separator />
                  <Box w="full">
                    <Text fontWeight="medium">👩‍🏫 Lehrer Feedback</Text>
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
}
