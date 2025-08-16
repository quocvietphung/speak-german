"use client";
import React from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  Heading,
  Textarea,
  Card,
  Badge,
  VisuallyHidden,
  Separator,
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { MdMic, MdStop } from "react-icons/md";

// Props for the RecordingCard component
interface RecordingCardProps {
  targetText: string; // The sentence the user should practice
  recording: boolean; // Whether recording is in progress
  transcript: string; // The recognized transcript from the user's recording
  onNextSentence: () => void; // Handler to fetch the next sentence
  onRecord: () => void; // Handler to start/stop recording
}

// Keyframes for the animated pulse effect around the record button
const pulse = keyframes`
  0% { transform: scale(1); opacity: .7; }
  70% { transform: scale(1.8); opacity: 0; }
  100% { transform: scale(1.8); opacity: 0; }
`;

// Keyframes for the animated equalizer bars
const bounce1 = keyframes`0%,100%{height:8px} 50%{height:28px}`;
const bounce2 = keyframes`0%,100%{height:16px} 50%{height:32px}`;
const bounce3 = keyframes`0%,100%{height:10px} 50%{height:26px}`;

// Main component for pronunciation practice
export default class RecordingCard extends React.Component<RecordingCardProps> {
  render() {
    // Destructure props for easier access
    const { targetText, recording, transcript, onNextSentence, onRecord } = this.props;

    return (
      <Card.Root
        p={6}
        rounded="2xl"
        shadow="xl"
        borderWidth="1px"
        bgGradient="linear(to-br, white, gray.50)"
      >
        {/* Header section with title and subtitle */}
        <Card.Header mb={4}>
          <VStack align="start">
            <Heading size="md">🎤 Pronunciation Practice</Heading>
            <Text fontSize="sm" color="gray.500">
              Record • Compare • Improve
            </Text>
          </VStack>
        </Card.Header>

        {/* Main body with controls and displays */}
        <Card.Body>
          <VStack align="center" gap={6} w="full">
            {/* Target sentence display */}
            <Box textAlign="center">
              <Badge colorScheme="blue" rounded="full" px={3} py={1}>
                Target
              </Badge>
              <Text mt={3} fontSize="xl" fontWeight="semibold" color="gray.800">
                {targetText}
              </Text>
            </Box>

            {/* Audio recording controls */}
            <VStack gap={3}>
              <Box position="relative" w="96px" h="96px">
                {/* Animated pulse effect when recording is active */}
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
                {/* Record/Stop button, toggles recording state */}
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
                  {/* Icon changes depending on recording state */}
                  {recording ? <MdStop /> : <MdMic />}
                </Button>
                {/* Visually hidden label for accessibility */}
                <VisuallyHidden>
                  <span>{recording ? "Recording…" : "Start recording"}</span>
                </VisuallyHidden>
              </Box>

              {/* Animated equalizer bars, active when recording */}
              <HStack gap={2} h="30px" align="end">
                {[bounce1, bounce2, bounce3].map((anim, i) => (
                  <Box
                    key={i}
                    w="10px"
                    rounded="sm"
                    bg={recording ? "purple.400" : "gray.300"}
                    animation={recording ? `${anim} 1s ease-in-out ${i * 0.1}s infinite` : undefined}
                  />
                ))}
              </HStack>
              {/* Status text below the controls */}
              <Text fontSize="sm" color="gray.600">
                {recording ? "🎙️ Recording…" : "Tap mic to start"}
              </Text>
            </VStack>

            {/* Horizontal separator line */}
            <Separator orientation="horizontal" />

            {/* Button to fetch the next practice sentence */}
            <Button
              onClick={onNextSentence}
              colorScheme="teal"
              variant="solid"
              size="md"
              w="full"
              rounded="xl"
              shadow="md"
            >
              ➡️ Next Sentence
            </Button>

            {/* Transcript display area, shows recognized speech */}
            <Box w="full">
              <Text fontWeight="medium" mb={2}>
                Transcript
              </Text>
              <Textarea
                value={transcript}
                readOnly
                rows={4}
                placeholder="Transcript will appear here..."
                bg="gray.50"
                borderColor="gray.200"
                rounded="lg"
                shadow="sm"
                _focus={{ borderColor: "blue.400", shadow: "md" }}
                fontFamily="ui-monospace, monospace"
              />
            </Box>
          </VStack>
        </Card.Body>
      </Card.Root>
    );
  }
}