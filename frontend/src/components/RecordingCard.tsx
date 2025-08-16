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
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { MdMic, MdStop } from "react-icons/md";

interface RecordingCardProps {
  targetText: string;
  recording: boolean;
  transcript: string;
  onNextSentence: () => void;
  onRecord: () => void;
}

/** Material-like pulse rings behind the mic when recording */
const pulse = keyframes`
  0% { transform: scale(1); opacity: .6; }
  70% { transform: scale(1.8); opacity: 0; }
  100% { transform: scale(1.8); opacity: 0; }
`;

/** Simple equalizer bars */
const bounce1 = keyframes` 0%,100%{height:8px} 50%{height:22px} `;
const bounce2 = keyframes` 0%,100%{height:14px} 50%{height:28px} `;
const bounce3 = keyframes` 0%,100%{height:10px} 50%{height:24px} `;

export default class RecordingCard extends React.Component<RecordingCardProps> {
  render() {
    const { targetText, recording, transcript, onNextSentence, onRecord } = this.props;

    return (
      <Card.Root p={6} rounded="2xl" shadow="sm" borderWidth="1px" bg="white">
        <Card.Header>
          <VStack align="start" gap={1}>
            <Heading size="md">🎤 Pronunciation Practice</Heading>
            <Text fontSize="sm" color="gray.500">
              Material-inspired controls • clear motion & states
            </Text>
          </VStack>
        </Card.Header>

        <Card.Body>
          <VStack align="start" gap={6} w="full">
            {/* Target sentence */}
            <Box>
              <Badge
                variant="subtle"
                colorScheme="blue"
                rounded="full"
                px={3}
                py={1}
              >
                Target
              </Badge>
              <Text mt={2} fontSize="xl" fontWeight="semibold" lineHeight="1.4">
                {targetText}
              </Text>
            </Box>

            {/* Controls */}
            <HStack gap={4} align="center">
              {/* Mic button with pulse ring (like M3 FAB state layer) */}
              <Box position="relative" w="72px" h="72px">
                {/* Pulsing rings when recording */}
                {recording && (
                  <>
                    <Box
                      position="absolute"
                      inset="0"
                      rounded="full"
                      bg="red.400"
                      opacity={0.25}
                      animation={`${pulse} 1.6s ease-out infinite`}
                    />
                    <Box
                      position="absolute"
                      inset="0"
                      rounded="full"
                      bg="red.400"
                      opacity={0.18}
                      animation={`${pulse} 1.6s .4s ease-out infinite`}
                    />
                  </>
                )}

                {/* Primary mic button */}
                <Button
                  onClick={onRecord}
                  aria-pressed={recording}
                  aria-label={recording ? "Stop recording" : "Start recording"}
                  title={recording ? "Stop" : "Record"}
                  rounded="full"
                  w="72px"
                  h="72px"
                  p={0}
                  fontSize="28px"
                  shadow={recording ? "lg" : "md"}
                  _hover={{ transform: "translateY(-1px)" }}
                  _active={{ transform: "translateY(0)" }}
                  transition="all 120ms"
                  colorScheme={recording ? "red" : "blue"}
                >
                  {recording ? <MdStop /> : <MdMic />}
                </Button>
                <VisuallyHidden>
                  <span>{recording ? "Recording on" : "Recording off"}</span>
                </VisuallyHidden>
              </Box>

              {/* Live state & mini equalizer */}
              <VStack align="start" gap={1}>
                <Text fontSize="sm" color="gray.600">
                  {recording ? "Recording…" : "Tap the mic to start"}
                </Text>
                <HStack gap={1.5} h="28px" align="end">
                  {[bounce1, bounce2, bounce3].map((anim, i) => (
                    <Box
                      key={i}
                      w="6px"
                      rounded="sm"
                      bg={recording ? "red.400" : "gray.300"}
                      animation={recording ? `${anim} 900ms ease-in-out ${i * 0.08}s infinite` : "none"}
                    />
                  ))}
                </HStack>
              </VStack>

              {/* Next sentence */}
              <Button
                onClick={onNextSentence}
                colorScheme="teal"
                variant="solid" // filled / emphasis, akin to M3 filled button
                size="sm"
              >
                Next sentence
              </Button>
            </HStack>

            {/* Transcript */}
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
                _focus={{ borderColor: "blue.300", shadow: "sm" }}
                fontFamily="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace"
              />
            </Box>
          </VStack>
        </Card.Body>
      </Card.Root>
    );
  }
}
