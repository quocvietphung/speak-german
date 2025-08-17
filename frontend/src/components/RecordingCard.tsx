"use client";
import React from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  Heading,
  Card,
  Badge,
  VisuallyHidden,
  Separator,
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { MdMic, MdStop } from "react-icons/md";

interface RecordingCardProps {
  targetText: string;
  recording: boolean;
  onNextSentence: () => void;
  onRecord: () => void;
}

const pulse = keyframes`
  0% { transform: scale(1); opacity: .7; }
  70% { transform: scale(1.8); opacity: 0; }
  100% { transform: scale(1.8); opacity: 0; }
`;
const bounce1 = keyframes`0%,100%{height:8px} 50%{height:28px}`;
const bounce2 = keyframes`0%,100%{height:16px} 50%{height:32px}`;
const bounce3 = keyframes`0%,100%{height:10px} 50%{height:26px}`;

export default class RecordingCard extends React.Component<RecordingCardProps> {
  render() {
    const { targetText, recording, onNextSentence, onRecord } = this.props;

    return (
      <Card.Root
        p={6}
        rounded="2xl"
        shadow="xl"
        borderWidth="1px"
        bgGradient="linear(to-br, white, gray.50)"
      >
        <Card.Header mb={4}>
          <VStack align="start">
            <Heading size="md">🎤 Pronunciation Practice</Heading>
            <Text fontSize="sm" color="gray.500">
              Record • Compare • Improve
            </Text>
          </VStack>
        </Card.Header>

        <Card.Body>
          <VStack align="center" gap={6} w="full">
            {/* Target sentence */}
            <Box textAlign="center">
              <Badge colorPalette="blue" rounded="full" px={3} py={1}>
                Target
              </Badge>
              <Text mt={3} fontSize="xl" fontWeight="semibold" color="gray.800">
                {targetText}
              </Text>
            </Box>

            {/* Record controls */}
            <VStack gap={3}>
              <Box position="relative" w="96px" h="96px">
                {recording && (
                  <>
                    <Box
                      position="absolute"
                      inset="0"
                      rounded="full"
                      bg="pink.400"
                      opacity={0.3}
                      animation={`${pulse} 1.8s ease-out infinite`}
                    />
                    <Box
                      position="absolute"
                      inset="0"
                      rounded="full"
                      bg="purple.400"
                      opacity={0.25}
                      animation={`${pulse} 2s 0.5s ease-out infinite`}
                    />
                  </>
                )}

                <Button
                  onClick={onRecord}
                  aria-pressed={recording}
                  aria-label={recording ? "Stop recording" : "Start recording"}
                  title={recording ? "Stop" : "Record"}
                  rounded="full"
                  w="96px"
                  h="96px"
                  p={0}
                  fontSize="36px"
                  bgGradient={
                    recording
                      ? "linear(to-br, red.400, pink.400)"
                      : "linear(to-br, blue.400, teal.400)"
                  }
                  color="white"
                  shadow="2xl"
                  transition="transform 0.2s"
                  _hover={{ transform: "scale(1.05)" }}
                  _active={{ transform: "scale(0.95)" }}
                >
                  {recording ? <MdStop /> : <MdMic />}
                </Button>

                <VisuallyHidden>
                  <span>{recording ? "Recording…" : "Start recording"}</span>
                </VisuallyHidden>
              </Box>

              {/* Equalizer */}
              <HStack gap={2} h="30px" align="end">
                {[bounce1, bounce2, bounce3].map((anim, i) => (
                  <Box
                    key={i}
                    w="10px"
                    rounded="sm"
                    bg={recording ? "purple.400" : "gray.300"}
                    animation={
                      recording ? `${anim} 1s ease-in-out ${i * 0.1}s infinite` : undefined
                    }
                  />
                ))}
              </HStack>

              <Text fontSize="sm" color="gray.600">
                {recording ? "🎙️ Recording…" : "Tap mic to start"}
              </Text>
            </VStack>

            <Separator orientation="horizontal" />

            {/* Next sentence (Transcript đã chuyển sang ScoreFeedbackCard) */}
            <Button
              onClick={onNextSentence}
              colorPalette="teal"
              variant="solid"
              size="md"
              w="full"
              rounded="xl"
              shadow="md"
            >
              ➡️ Next Sentence
            </Button>
          </VStack>
        </Card.Body>
      </Card.Root>
    );
  }
}
